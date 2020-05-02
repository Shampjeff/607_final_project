# Final Thoughts
This is a summary of some key take aways I had from completing this final project. 


## IDE
After a semester of using RStudio and Rmarkdown files to both code and write the assignment narrative, I more clearly see the advantages and disadvantages of Jupyter and Rstudio. These seem like the most popular IDEs (PyCharm gaining). The R community definitey has a better UX and final presentation than Jupyter. 


## Data Science Life-Cycle
This was a cool aspect of the project - taking a idea from data extraction all the way to model deployment was fun. Adding an extra level of using the model in the cloud for a web app was especially challenging but also rewarding. 


I like the unified tools that are available in python for data exploration, modeling, and deployment or maybe I'm just more familiar with python. I think the recent production of TidyModels will make this much easier in R and as well. I'm excited to use R more for full-cycle data science work. 


## AWS
This project has kind of soured me on using AWS. It really is the web service mirror of Amazon the retail site. An endless array of products, so many that it's impossible to know them all. Layers and layers and layers of services and protocols to follow for what seems like a simple task. Also every service has pages and pages of documentation, but much of it is so very cryptic that is takes hours to understand and follow. AWS is the giant in this space, but I really hope that GCP, Databricks, IBM or some other service with a more unified and user-friendly interface will dethrone AWS in the future. I have a few examples below of things that I learned from this project


### SageMaker
This is the AWS flagship ML service. After using it extensively on this project and from another online course, I have come to the conclusion that, while SageMaker has many nice features, it is not actually needed. If you _need_ the computing power then it is much easier to use than EC2, but that's about it. SageMaker is very costly and there are several cheaper, and sometimes easier, products that can accomplish the goals of SageMaker. 
1. EC2 can launch Jupyter Lab and it uses the same computing resources "under the hood" as SageMaker.
2. Model endpoints built in Sagemaker are not actually needed. A user can simply make a series of AWS Lambda layers with a final layer that loads a pre-trained model from a pickle file for inference. Lambda is **one-tenth** the cost of SageMaker. 
3. The secruity configuration for SageMaker and model endpoints is Byzantine at best and this can largely be avoided using other services (see above). 


### Lambda
Lambda is a serveless code execution service. It is meant to be lightweight and quick, which it is. However, Lambda cannot do the kind of pre-processing needed to handle models that use predict based on text. For example, loading Pandas and SKLearn libraries on lambda was not possible for me. Version, file size, dependencies were too complicated to use and the documentation for how to load libraries was very poor. 


## Web App
This part was tough. Very tough. But it was great to finally get an elementary site up and have it return real-time predictions on input text! For AWS security reasons, I was not able to host it live for others to use - that is an entirely different project, I think. 

I feel like I got a good overview of how to run a dynamic web application using a combination of python and html, which I had zero knowledge of coming into this project. 