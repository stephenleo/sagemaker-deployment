"""Example workflow pipeline script for heartdisease pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput
)
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.tuner import (
    HyperparameterTuner, 
    IntegerParameter, 
    ContinuousParameter, 
    CategoricalParameter
)

from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TuningStep,
    CacheConfig,
)
from sagemaker.workflow.model_step import ModelStep


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="HeartDiseasePackageGroup",
    pipeline_name="HeartDiseasePipeline",
    base_job_prefix="HeartDisease",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working on heartdisease data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    # Session information
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    pipeline_session = get_pipeline_session(region, default_bucket)

    # Parameters for pipeline execution
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", 
        default_value=1
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", 
        default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    )
    
    # Cache Pipeline steps to reduce execution time on subsequent executions
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")

    # Step 1: Processing step
    ## Create the processor
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-heartdisease-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )

    ## Create the pipeline step
    step_process = ProcessingStep(
        name="PreprocessHeartDiseaseData",
        processor=sklearn_processor,
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--input_data", input_data],
        cache_config=cache_config
    )

    # Step 2: Model training step   
    ## Create the estimator
    sklearn_estimator = SKLearn(
        base_job_name=f"{base_job_prefix}/heartdisease-train",
        framework_version="1.0-1",
        entry_point=os.path.join(BASE_DIR, "train.py"),
        instance_count=1,
        instance_type=training_instance_type,
        sagemaker_session=pipeline_session,
        role=role
    )
    
    # Define exploration boundaries
    hyperparameter_ranges = {
        "n_estimators": IntegerParameter(1, 20),
        "min_samples_split": ContinuousParameter(0.01, 0.5),
        "criterion": CategoricalParameter(["gini", "entropy"])
    }

    # Create optimizer
    optimizer = HyperparameterTuner(
        base_tuning_job_name="rfc-pipeline-tuner",
        estimator=sklearn_estimator,
        hyperparameter_ranges=hyperparameter_ranges,
        objective_type="Maximize",
        objective_metric_name="test-accuracy",
        metric_definitions=[
            {"Name": "train-accuracy", "Regex": "Training Accuracy: ([0-9.]+).*$"},
            {"Name": "test-accuracy", "Regex": "Testing Accuracy: ([0-9.]+).*$"}
        ],
        max_jobs=10,
        max_parallel_jobs=2,
    )

    ## Create the pipeline step
    step_train = TuningStep(
        name = "TrainHeartDiseaseModelTuning",
        tuner = optimizer,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config
    )

    # Step 3: Model evaluation step
    ## Create the processor
    sklearn_evaluator = SKLearnProcessor(
        framework_version="1.0-1",
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/sklearn-heartdisease-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )

    ## Create the pipeline step  
    evaluation_report = PropertyFile(
        name="HeartDiseaseEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    
    step_eval = ProcessingStep(
        name="EvaluateHeartDiseaseModel",
        processor=sklearn_evaluator,
        inputs=[
            ProcessingInput(
                source=step_train.get_top_model_s3_uri(top_k=0, s3_bucket=default_bucket),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", 
                source="/opt/ml/processing/evaluation"
            ),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report]
    )    
    
    # Step 4: Register model step that will be conditionally executed
    s3_uri = step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
    
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f'{s3_uri}/evaluation.json',
            content_type="application/json"
        )
    )
    
    model = SKLearnModel(
        framework_version="1.0-1",
        entry_point=os.path.join(BASE_DIR, "serve.py"),
        model_data=step_train.get_top_model_s3_uri(top_k=0, s3_bucket=default_bucket),
        sagemaker_session=pipeline_session,
        role=role,
    )
    
    step_args = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics
    )
    
    step_register = ModelStep(
        name="RegisterHeartDiseaseModel",
        step_args=step_args,
    )

    # condition step for evaluating model quality and branching execution
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value"
        ),
        right=0.80,
    )
    
    step_cond = ConditionStep(
        name="HeartDiseaseAccuracyCheck",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )

    # Step 5: Create the entire pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    
    return pipeline