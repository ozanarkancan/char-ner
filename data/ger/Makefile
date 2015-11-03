
%.bio: NER-de-%.tsv
	grep -v "^#" $< | awk '{print $$2"\t"$$4"\t"$$3}' > $@
