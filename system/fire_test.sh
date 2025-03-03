python -u main.py -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 100 -data Cifar100 -m CNNP-5 -fd 512 -did 0 -algo FedProto -lam 10

python -u main.py -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 100 -data Cifar100 -m CNN-5 -fd 512 -did 0 -algo FedDMG -lam 10

