# **Machine Learning Model Deployment in the Cloud**
## **Deployment of ML models on the cloud using AWS Sagemaker**

**Author: Marie Stephen Leo**

### About the Course
This course aims to equip a learner with practical knowledge of Deploying ML models on the cloud using AWS Sagemaker.

### Target Audience
The target audience for this course would include professionals aspiring to become Data Scientists, Machine Learning Engineers, Data Analysts, Statisticians, Data Architects, Data engineers, etc. 

## Setup
1. Sign up for a free AWS account at https://aws.amazon.com
1. In your AWS console search for “sagemaker” and click it

    <img src="images/1.png" width="1000">

1. Click on Studio

    <img src="images/2.png" width="750">

1. Click on Launch SageMaker Studio

    <img src="images/3.png" width="750">

1. The first time you launch Sagemaker Studio, you’ll have to do a one time setup. 
    - Select Quick setup 
    - Type a name that’s meaningful
    - Select create a new role from the dropdown

    <img src="images/4.png" width="1000">

1. Select “Any S3 bucket” so that your Studio notebooks have access to any data on any S3 bucket in your account. Click Create role

    <img src="images/5.png" width="750">

1. Once the role is created, click Submit

    <img src="images/6.png" width="1000">

1. The setup will take some time. You can see the status on the top of the page

    <img src="images/7.png" width="750">

1. Once the setup is done, you’ll see the domain you created with a “Launch app” dropdown
Click on “Studio”. **DO NOT CLICK ON CANVAS ($$$)!!!!**

    <img src="images/8.png" width="750">

1. After some loading, your brand new jupyter lab interface running on Sagemaker studio servers will open up. If the loading gets stuck on a white screen, try refreshing the page.
    - You can clone any github repository. For example, you can clone this course repository: https://github.com/stephenleo/sagemaker-deployment.git
    - Follow the lessons in the `notebooks/` directory.

    <img src="images/9.png" width="1000">
    
1. While running any code, if you get an error that the kernel is still starting, look at the bottom of your screen for the Kernel status

    <img src="images/10_1.png">
    
1. You can run the cell once the Kernel status is Idle

    <img src="images/10_2.png">

1. When you’re done with Sagemaker Studio, REMEMBER to shut down everything to prevent getting charged $!

    <img src="images/11.png">

    <img src="images/11_1.png">

1. You will get the below message only if you opened at least one jupyter notebook on the studio instance. So if you don’t get this, just create a new notebook, let it start up and then try shutting down again just to be sure.
    
    <img src="images/11_2.png">
