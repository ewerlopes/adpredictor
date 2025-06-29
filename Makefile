DEMO = python main.py \
	--num-examples 500 \
	--num-features 5 \
	--feature-cardinality 10 \
	--prior-probability 0.5 \
	--visualization-interval 1 \
	--biased-feature-proportion=0.02 \
	--biased-feature-effect-length 200

ANIMATE =  convert \
	-delay 50 \
	-resize 600000@ \
	-unsharp 0x1 \
	-loop 0 ./log/adpredictor/*.png

all: | test

proto:
	mkdir -p protobufs
	find . -iname '*.proto' | xargs -J % protoc --proto_path=. --python_out=protobufs %
	touch protobufs/__init__.py

freeze:
	pip freeze > requirements.txt

requirements:
	pip install -r requirements.txt

test:
	env/bin/nosetests

demo: | requirements proto
	rm -rf ./log/adpredictor/*
	mkdir -p ./log/adpredictor
	$(DEMO) --num-examples 25 --visualization-interval 1
	$(ANIMATE) ./log/initial_learning.gif

	rm -rf ./log/adpredictor/*
	$(DEMO) --num-examples 200 --visualization-interval 10
	$(ANIMATE) ./log/convergence_learning.gif

	rm -rf ./log/adpredictor/*
	$(DEMO) --num-examples 400 --visualization-interval 20
	$(ANIMATE) ./log/online_learning.gif

.PHONY:
	test
