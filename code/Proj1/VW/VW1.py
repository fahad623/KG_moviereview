vw rotten.train.vw -c -k --passes 300 --ngram 7 -b 24 --ect 5 -f rotten.model.vw
vw rotten.test.vw -t -i rotten.model.vw -p rotten.preds.txt

None 11 0.3418
1e-5 5  0.381
1e-7 12 0.3417
1e-8 11 0.3418
1e-9 12 0.341856

vw rotten.train.vw -c -k --passes 13 --ngram 7 -b 24 --ect 5 -f rotten.model.vw --holdout_off --l2 1e-7


vw rotten.train.vw -c -k --passes 300 --ngram 2 -b 24 --oaa 5 -f rotten.model.vw --loss_function logistic
vw rotten.test.vw -t -i rotten.model.vw -p rotten.preds.txt -r rotten.raw.txt