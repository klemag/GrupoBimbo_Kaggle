## Need a local copy of cookies.txt with cookies from Kaggle
sudo wget -x -c --load-cookies cookies.txt -P data -nH --cut-dirs=5 https://www.kaggle.com/c/grupo-bimbo-inventory-demand/download/cliente_tabla.csv.zip
sudo wget -x -c --load-cookies cookies.txt -P data -nH --cut-dirs=5 https://www.kaggle.com/c/grupo-bimbo-inventory-demand/download/producto_tabla.csv.zip
sudo wget -x -c --load-cookies cookies.txt -P data -nH --cut-dirs=5 https://www.kaggle.com/c/grupo-bimbo-inventory-demand/download/train.csv.zip
sudo wget -x -c --load-cookies cookies.txt -P data -nH --cut-dirs=5 https://www.kaggle.com/c/grupo-bimbo-inventory-demand/download/test.csv.zip
sudo wget -x -c --load-cookies cookies.txt -P data -nH --cut-dirs=5 https://www.kaggle.com/c/grupo-bimbo-inventory-demand/download/town_state.csv.zip

## Unzip
sudo unzip cliente_tabla.csv.zip
sudo unzip test.csv.zip 
sudo unzip town_state.csv.zip
sudo unzip train.csv.zip 
sudo unzip producto_tabla.csv.zip 
sudo rm *zip