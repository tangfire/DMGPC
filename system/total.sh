nohup python -u main.py -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtM10 -fd 512 -did 0 -algo Local > total-Cifar100-HtM10-fd=512-Local.out 2>&1 &
nohup python -u main.py -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtM10 -fd 512 -did 1 -algo FedProto -lam 10 > total-Cifar100-HtM10-fd=512-FedProto.out 2>&1 &
nohup python -u main.py -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtM10 -fd 512 -did 1 -algo FedGen -nd 32 -glr 0.1 -hd 512 -se 100 > total-Cifar100-HtM10-fd=512-FedGen.out 2>&1 &
nohup python -u main.py -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtM10 -fd 512 -did 2 -algo FedDistill -lam 1 > total-Cifar100-HtM10-fd=512-FedDistill.out 2>&1 &
nohup python -u main.py -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtM10 -fd 512 -did 2 -algo FML -al 0.5 -bt 0.5 > total-Cifar100-HtM10-fd=512-FML.out 2>&1 &
nohup python -u main.py -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtM10 -fd 512 -did 3 -algo FedKD -mlr 0.01 -Ts 0.95 -Te 0.98 > total-Cifar100-HtM10-fd=512-FedKD.out 2>&1 &
nohup python -u main.py -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtM10 -fd 512 -did 3 -algo LG-FedAvg > total-Cifar100-HtM10-fd=512-LG-FedAvg.out 2>&1 &
nohup python -u main.py -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtM10 -fd 512 -did 3 -algo FedGH -slr 0.01 > total-Cifar100-HtM10-fd=512-FedGH.out 2>&1 &
nohup python -u main.py -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtM10 -fd 512 -did 4 -algo FedTGP -lam 10 -se 100 -mart 100 > total-Cifar100-HtM10-fd=512-FedTGP.out 2>&1 &
nohup python -u main.py -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtM10 -fd 512 -did 5 -algo FedKTL-stylegan-xl -slr 0.01 -sbs 100 -se 100 -lam 1 -mu 50 > total-Cifar100-HtM10-fd=512-FedKTL-stylegan-xl.out 2>&1 &
nohup python -u main.py -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtM10 -fd 512 -did 2 -algo FedKTL-stylegan-3 -slr 0.01 -sbs 100 -se 100 -lam 0.1 -mu 50 -GPath stylegan/stylegan-3-models/Benches-512.pkl > total-Cifar100-HtM10-fd=512-FedKTL-stylegan-3.out 2>&1 &
nohup python -u main.py -t 3 -ab 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -nc 20 -nb 100 -data Cifar100 -m HtM10 -fd 512 -did 1 -algo FedKTL-stable-diffusion -slr 0.1 -sbs 100 -se 100 -lam 0.01 -mu 100 -GPath stable-diffusion/v1.5 > total-Cifar100-HtM10-fd=512-FedKTL-stable-diffusion.out 2>&1 &
