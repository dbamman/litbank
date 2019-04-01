cd trained_model

wget "http://tti-coin.jp/data/wikipedia200.bin"

python ../scripts/convert_embeddings_bin_to_txt.py wikipedia200.bin wikipedia200.txt

rm wikipedia200.bin
