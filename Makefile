
char.bi-relu.feat_%:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python exper.py --activation bi-relu --epoch 1 \
		     --hidden 50 --opt adam --norm 7 --bias 1 --fepoch 300 --feat $* > logs/$@.log

char.bi-relu.deep.h%:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python exper.py --activation bi-relu bi-relu --epoch 1 \
		     --hidden $* $* --opt adam --norm 7 --bias 1 1 --drates 0 0 0 --fepoch 300 --feat basic_seg > logs/$@.log

word.bi-relu:
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python exper_word.py --activation relu --epoch 1 \
		     --hidden 50 --opt adam --norm 7 --bias 1 --fepoch 300 > logs/$@.log
