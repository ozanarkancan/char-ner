

%.iob: deu.%
	tail -n+3 $< | grep -v DOCSTART | tr '\n' '_' | sed 's/___*/__/g' | tr '_' '\n' > $@
