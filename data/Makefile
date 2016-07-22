UNI_DIR=/ai/home/okuru13/research/data/universalDependencies/ud-treebanks-v1.3/

cze.pos.train:
	cat ${UNI_DIR}/UD_Czech/cs-ud-train.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > cze-pos/train.bio

cze.pos.testa:
	cat ${UNI_DIR}/UD_Czech/cs-ud-dev.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > cze-pos/testa.bio

cze.pos.testb:
	cat ${UNI_DIR}/UD_Czech/cs-ud-test.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > cze-pos/testb.bio

eng.pos.train:
	cat ${UNI_DIR}/UD_English/en-ud-train.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > eng-pos/train.bio

eng.pos.testa:
	cat ${UNI_DIR}/UD_English/en-ud-dev.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > eng-pos/testa.bio

eng.pos.testb:
	cat ${UNI_DIR}/UD_English/en-ud-test.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > eng-pos/testb.bio

tr.pos.train:
	cat ${UNI_DIR}/UD_Turkish/tr-ud-train.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > tr-pos/train.bio

tr.pos.testa:
	cat ${UNI_DIR}/UD_Turkish/tr-ud-dev.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > tr-pos/testa.bio

tr.pos.testb:
	cat ${UNI_DIR}/UD_Turkish/tr-ud-test.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > tr-pos/testb.bio

ned.pos.train:
	cat ${UNI_DIR}/UD_Dutch/nl-ud-train.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > ned-pos/train.bio

ned.pos.testa:
	cat ${UNI_DIR}/UD_Dutch/nl-ud-dev.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > ned-pos/testa.bio

ned.pos.testb:
	cat ${UNI_DIR}/UD_Dutch/nl-ud-test.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > ned-pos/testb.bio

deu.pos.train:
	cat ${UNI_DIR}/UD_German/de-ud-train.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > deu-pos/train.bio

deu.pos.testa:
	cat ${UNI_DIR}/UD_German/de-ud-dev.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > deu-pos/testa.bio

deu.pos.testb:
	cat ${UNI_DIR}/UD_German/de-ud-test.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > deu-pos/testb.bio

spa.pos.train:
	cat ${UNI_DIR}/UD_Spanish/es-ud-train.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > spa-pos/train.bio

spa.pos.testa:
	cat ${UNI_DIR}/UD_Spanish/es-ud-dev.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > spa-pos/testa.bio

spa.pos.testb:
	cat ${UNI_DIR}/UD_Spanish/es-ud-test.conllu | grep -v "^#" | grep -vP "^[0-9]+\-" | cut -f2,4 | awk '{if($$0!="") print $$1"\tB-"$$2; else print $$0}' > spa-pos/testb.bio
