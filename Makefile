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
	-loop 0 ./logs/adpredictor/*.png

all: | test

proto:
	mkdir -p protobufs
	find . -iname '*.proto' | xargs -J % protoc --proto_path=. --python_out=protobufs %
	touch protobufs/__init__.py

requirements:
	uv lock
	uv sync

test:
	env/bin/nosetests

demo: | requirements proto
	rm -rf ./logs/adpredictor/*
	mkdir -p ./logs/adpredictor
	$(DEMO) --num-examples 25 --visualization-interval 1
	$(ANIMATE) ./logs/initial_learning.gif

	rm -rf ./logs/adpredictor/*
	$(DEMO) --num-examples 200 --visualization-interval 10
	$(ANIMATE) ./logs/convergence_learning.gif

	rm -rf ./logs/adpredictor/*
	$(DEMO) --num-examples 400 --visualization-interval 20
	$(ANIMATE) ./logs/online_learning.gif

.PHONY:
	test
