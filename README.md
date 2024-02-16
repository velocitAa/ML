An example of how to deploy Machine learning model using Fast API and Docker
How to run
Install and run Docker
Build Docker image using docker build . -t beer_server
Run Docker container using docker run --rm -it -p 8000:8000 beer_server
Go to http://127.0.0.1:8000/docs to see all available methods of the API

Source code
Rec
main.py contains API logic (FastApi + model)
data_for_cosine_similarity.csv file where first column is possible types of product for which you can get recommendation 
Dockerfile describes a Docker image that is used to run the API
P.S. if name of beer contain empty space " " change it to "_" example: Amstel Light -> Amstel_Light. Also GET method works only with string example input: "Budweiser Amstel_Light Bud_Light"
Jupyter notebook Rec_system.ipynb contains three models (SVD, ItemKNN and RSLM (Pytorch))
