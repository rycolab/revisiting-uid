

DATA_URL := https://raw.githubusercontent.com/languageMIT/naturalstories/master/
READING_TIMES_URL := $(DATA_URL)naturalstories_RTS/processed_RTs.tsv
SURPRISALS_URL := $(DATA_URL)probs/all_stories_gpt3.csv


DATA_PATH := ./data
READING_TIMES_FILE := $(DATA_PATH)/reading_times.tsv
SURPRISALS_FILE := $(DATA_PATH)/surprisals.csv

all: get_data

get_data: $(READING_TIMES_FILE) $(SURPRISALS_FILE)

$(SURPRISALS_FILE):
	mkdir -p $(DATA_PATH)
	wget $(SURPRISALS_URL) -O $(SURPRISALS_FILE)

$(READING_TIMES_FILE):
	mkdir -p $(DATA_PATH)
	wget $(READING_TIMES_URL) -O $(READING_TIMES_FILE)
