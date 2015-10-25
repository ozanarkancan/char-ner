
%.bio: %.iob
	cat $< | tr '\t' ' ' | perl ../../scripts/chrep.pl iob2 | tr ' ' '\t' > $@
