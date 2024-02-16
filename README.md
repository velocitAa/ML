1. An example of how to deploy Machine learning model using Fast API and Docker
2. How to run
3. Install and run Docker
4. Build Docker image using docker build . -t beer_server
5. Run Docker container using docker run --rm -it -p 8000:8000 beer_server
6. Go to http://127.0.0.1:8000/docs to see all available methods of the API
---
Source code
1. main.py contains API logic (FastApi + model)
2. data_for_cosine_similarity.csv file where first column is possible types of product for which you can get recommendation 
3. Dockerfile describes a Docker image that is used to run the API
> P.S. if name of beer contains empty space " " change it to "_" example: Amstel Light -> Amstel_Light. Also GET method works only with string example input: "Budweiser Amstel_Light Bud_Light"
4. Jupyter notebook Rec_system.ipynb contains three models (SVD, ItemKNN and RSLM (Pytorch))
---
Full project you can download from here: https://drive.google.com/file/d/1lEuAAuQSZ_LqFV698rr26_Tx1KhiL68P/view?usp=sharing
