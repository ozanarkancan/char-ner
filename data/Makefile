UNI_DIR=/ai/home/okuru13/research/data/universalDependencies/ud-treebanks-v1.3/

cze.pos.train:
	cat ${UNI_DIR}/UD_Czech/cs-ud-train.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > cze-pos/train.bio

cze.pos.testa:
	cat ${UNI_DIR}/UD_Czech/cs-ud-dev.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > cze-pos/testa.bio

cze.pos.testb:
	cat ${UNI_DIR}/UD_Czech/cs-ud-test.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > cze-pos/testb.bio
