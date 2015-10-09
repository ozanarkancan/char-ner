

%.bio: ned.%
	tail -n+2 $< | grep -v DOCSTART | tr '\n' '_' | sed 's/___*/__/g' | tr '_' '\n' | tr ' ' '\t' > $@
