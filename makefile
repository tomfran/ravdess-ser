# utility to zip src and processed to use them in google colab

zip_src: 
	zip -r src.zip src/ -x '*__pycache__*'

zip_data:
	zip -r processed.zip data/processed/

zip_all: 
	make zip_src && make zip_data