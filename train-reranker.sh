#!/usr/bin/env bash

# prepare vocab
python prepare_vocab.py --config_dir=resources/config/feature_column --vocab_dir=resources/vocab/

# prepare train/eval data
python prepare_dataset.py --train_dir=resources/train --eval_dir=resources/eval --config_dir=resources/config/feature_column --train_part_num=10


# start rerank model training
if [[ ! -d "logs" ]]
then
    mkdir logs
fi

if [[ -d "rerank_model" ]]
then
    rm -rf rerank_model
fi

time=`date +%s`
# chief
TF_CONFIG='{"cluster": {"chief": ["localhost:2222"], "worker": ["localhost:2223"], "ps": ["localhost:2226"], "evaluator": ["localhost:2227"]}, "task": {"type": "chief", "index": 0}}' nohup python reranker_trainer.py --config_dir=resources/config --vocab_dir=resources/vocab --train_files_dir=resources/train --eval_files_dir=resources/eval --model_path=rerank_model --use_float16=True --enable_xla=False --train_max_steps=1000 --checkpoint_steps=200 >logs/chief-$time 2>&1 &
chief_pid=`echo $!`
echo "chief pid: $chief_pid"

# ps
TF_CONFIG='{"cluster": {"chief": ["localhost:2222"], "worker": ["localhost:2223"], "ps": ["localhost:2226"], "evaluator": ["localhost:2227"]}, "task": {"type": "ps", "index": 0}}' nohup python reranker_trainer.py --config_dir=resources/config --vocab_dir=resources/vocab --train_files_dir=resources/train --eval_files_dir=resources/eval --model_path=rerank_model --use_float16=True --enable_xla=False --train_max_steps=1000 --checkpoint_steps=200 >logs/ps-$time 2>&1 &
ps_pid=`echo $!`
echo "ps pid: $ps_pid"

# worker
TF_CONFIG='{"cluster": {"chief": ["localhost:2222"], "worker": ["localhost:2223"], "ps": ["localhost:2226"], "evaluator": ["localhost:2227"]}, "task": {"type": "worker", "index": 0}}' nohup python reranker_trainer.py --config_dir=resources/config --vocab_dir=resources/vocab --train_files_dir=resources/train --eval_files_dir=resources/eval --model_path=rerank_model --use_float16=True --enable_xla=False --train_max_steps=1000 --checkpoint_steps=200 >logs/worker-$time 2>&1 &
worker_pid=`echo $!`
echo "worker pid: $worker_pid"

# evaluator
TF_CONFIG='{"cluster": {"chief": ["localhost:2222"], "worker": ["localhost:2223"], "ps": ["localhost:2226"], "evaluator": ["localhost:2227"]}, "task": {"type": "evaluator", "index": 0}}' python reranker_trainer.py --config_dir=resources/config --vocab_dir=resources/vocab --train_files_dir=resources/train --eval_files_dir=resources/eval --model_path=rerank_model --use_float16=True --enable_xla=False --train_max_steps=1000 --checkpoint_steps=200 >logs/evaluator-$time 2>&1 &
evaluator_pid=`echo $!`
echo "evaluator pid: $evaluator_pid"

signal_handler() {
    kill -9 $chief_pid
    kill -9 $ps_pid
    kill -9 $worker_pid
    kill -9 $evaluator_pid
}
trap signal_handler INT

# check chief running
while kill -0 $chief_pid 2> /dev/null; do
    sleep 10
done

# check worker running
while kill -0 $chief_pid 2> /dev/null; do
    sleep 10
done

# check evaluator running
while kill -0 $evaluator_pid 2> /dev/null; do
    sleep 10
done

# kill ps
kill $ps_pid
wait
