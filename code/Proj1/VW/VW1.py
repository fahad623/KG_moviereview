vw rotten.train.vw -c -k --passes 300 --ngram 2 -b 24 --ect 5 -f rotten.model.vw
vw rotten.test.vw -t -i rotten.model.vw -p rotten.preds.txt


vw rotten.train.vw -c -k --passes 300 --ngram 2 -b 24 --oaa 5 -f rotten.model.vw --loss_function logistic
vw rotten.test.vw -t -i rotten.model.vw -p rotten.preds.txt -r rotten.raw.txt