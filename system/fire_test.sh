python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 100 -data Cifar100 -m CNNP-5 -fd 512 -did 0 -algo FedProto -lam 10

python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 100 -data Cifar100 -m CNN-5 -fd 512 -did 0 -algo FedDMG -lam 10

python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 100 -data Cifar100 -m CNNP-5 -fd 512 -did 0 -algo FedMRL -lam 10

python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 100 -data Cifar100 -m CNNP-5 -fd 512 -did 0 -algo FedTGP -lam 10

python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 100 -data Cifar100 -m CNNP-5 -fd 512 -did 0 -algo FedGH -lam 10

python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 100 -data Cifar100 -m SCNN-4 -fd 512 -did 0 -algo FedProto -lam 10

python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 100 -data Cifar100 -m HCNN-4 -fd 512 -did 0 -algo FedDMGV2 -lam 10

python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 100 -data Cifar100 -m SCNN-4 -fd 512 -did 0 -algo FedMRL -lam 10


python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 100 -data Cifar100 -m SCNN-4 -fd 512 -did 0 -algo FedTGP -lam 10

python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -ncl 100 -data Cifar100 -m HCNN-4 -fd 512 -did 0 -algo FedDMGTGP -lam 10
