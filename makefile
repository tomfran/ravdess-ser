# utility to zip src and upload it to Google Colab

zip_src: 
	zip -r src.zip src/ -x '*__pycache__*'