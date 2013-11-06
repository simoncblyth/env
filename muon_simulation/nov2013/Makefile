PYTHON?=python

SLIDES=$(subst .txt,.html,$(wildcard *.txt))

slides: $(SLIDES)

%.html: %.txt
	$(PYTHON) ./rst2s5-2.6.py --theme small-white --current-slide --visible-controls --language=en $< $@

clean:
	rm -f $(SLIDES)
