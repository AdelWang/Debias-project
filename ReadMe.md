# Debias-project
project
run_glue_self.py  在huggingface transformers 的 /example/pytorch/text-classification 下
trainer_self.py   在对应库中，与trainer.py在同一路径
modeling_bert_cl.py 在transformers库中 /model/bert/
路径下的 __init__.py 也import对应的py文件

data_process.py 对原数据进行预处理，计算句子对间的overlap，并写入成csv （huggingface在导入时需要训练集和dev集保持一致，所以也计算了dev集句子对的overlap）
lemme是指调用NLTK中的lemmenizer将一些词的变位还原，如 apples --> apple,  gone --> go，还原后再计算overlap。



```
python run_glue_self.py \
	--num_train_epochs 3 \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --do_train \
    --do_eval \
    --model_name_or_path ./PLM/bert_base/ \
    --train_file ./data/mnli/train_lemme.csv \
    --validation_file ./data/mnli/dev_matched_overlap.csv \
    --learning_rate 2e-5 \
    --output_dir ./CL/ \
    ## use "--use_classifier_pred False" to enable the Adv training
    '''
    --learning_rate_classifier 5e-4 \ 
    --weight_mse 2 \
    --activation relu \
    '''
    --max_patience 100 \
    --smoother True \ ## using the smoothing mechanism
    --layer 35 \ ## 作CL的层数，layer目前仅限两位数
    --cl_type token \ ## choice = {token, sentence} \
    --temp 0.1 \
    --weight_cl 0.1 
```



Contrastive Learning on MNLI and STS-B

参数设定为3-5 对应4-6层，MNLI：lr = 2e-5, STS-B:  lr = 4e-5

| temp/weight | MNLI m/mm (On dev) | STS-B (On test) <br />P/S correlation |
| :---------: | :----------------: | :-----------------------------------: |
|   Without   |     84.6/84.8      |             86.03 / 85.00             |
|  0.1/0.01   |     84.9/85.2      |             86.13 / 85.14             |
|  0.1/0.05   |     84.5/84.8      |             86.39 / 85.32             |
|   0.1/0.1   |    84.8 / 84.7     |             86.28 / 85.28             |



Adv training on HANS

HANS (MNLI): lr = 5e-5

PAWS (QQP): lr = 2e-5

|             lr_classifier / weight              | HANS | PAWS  |
| :---------------------------------------------: | :--: | :---: |
|                     without                     | 64.6 | 35.0  |
|                    1e-3 / 1                     | 69.5 | 33-34 |
|          + Smoothing mechanism (0.45)           |      |       |
|  smoothing = epoch / nums_epoch<br />5e-4 / 2   | 68.3 |       |
| smoothing = self.sigmoid(epoch,1)<br />5e-4 / 2 | 68.6 |       |

MNLI

![img](https://docimg9.docs.qq.com/image/If_qn-Y_OzfFuY4I5jbL8Q.png?w=462&h=304)

QQP

![img](https://docimg8.docs.qq.com/image/WVyos4x8XRIg5bvu6y3RHQ.png?w=529&h=342)

