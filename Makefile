
HOST=$(shell hostname)
ifeq (${HOST}, balina.ku.edu.tr)
    LOGDIR=/ai/home/okuru13/research/ner/seq/run/char-ner/logs
    DEVICE=cpu
else ifeq (${HOST}, altay.ku.edu.tr)
    LOGDIR=/ai/home/okuru13/research/ner/seq/run/char-ner/logs
    DEVICE=cpu
else ifeq (${HOST}, ural.ku.edu.tr)
    LOGDIR=/ai/home/okuru13/research/ner/seq/run/char-ner/logs
    DEVICE=cpu
else
    LOGDIR=/mnt/kufs/scratch/okuru13/logs
    DEVICE=gpu
endif

char.bi-relu.feat_%:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python exper.py --activation bi-relu --epoch 1 \
		     --hidden 50 --opt adam --norm 7 --bias 1 --fepoch 300 --feat $* > logs/$@.log

char.bi-relu.deep.h%:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python exper.py --activation bi-relu bi-relu --epoch 1 \
		     --hidden $* $* --opt adam --norm 7 --bias 1 1 --drates 0 0 0 --fepoch 300 --feat basic_seg > logs/$@.log

word.bi-relu:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python exper_word.py --activation relu --epoch 1 \
		     --hidden 50 --opt adam --norm 7 --bias 1 --fepoch 300 > logs/$@.log
laz.char.bi-relu.h%:
	THEANO_FLAGS=mode=FAST_RUN,device=${DEVICE},floatX=float32 python exper_lasagne.py --activation rectify\
		     --n_hidden $* --opt adam --grad_clip 7 --fepoch 300 > ${LOGDIR}/$@.log

laz.char.bi-lstm.h%:
	THEANO_FLAGS=mode=FAST_RUN,device=${DEVICE},floatX=float32 python exper_lasagne.py --activation rectify\
		     --ltype lstm --n_hidden $* --opt adam --grad_clip 7 --fepoch 300
test:
	echo ${DEVICE}
