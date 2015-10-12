
train.bio: dse.txt
	cat $< | awk '{\
	    if($$0=="") print $$0;\
	    else{\
	    if($$3=="B_DSE")t="B-DSE"; else if($$3=="DSE")t="I-DSE"; else t="O"; print $$1"\t"$$2"\t"t}}' > $@
