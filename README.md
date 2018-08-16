## A _baseline_ model for [Zalo Landmark Identification challenge](https://challenge.zalo.ai/portal/overview)

## Get code
```
git clone https://github.com/tiepvupsu/zalo_landmark
cd zalo_landmark
```

## Train 

```
cd src
python train.py
```

## Test 
```
python inference.py --model_path=<date and time of the saved model>
```

## Results

Top 3 error on the Public dataset: 0.02054
