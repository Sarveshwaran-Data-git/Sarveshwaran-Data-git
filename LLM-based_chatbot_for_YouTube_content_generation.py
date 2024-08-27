{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2051de4f-d064-4864-99d3-0509188e9a8c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: transformers in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (4.44.2)\n",
      "Requirement already satisfied: torch in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (2.4.0)\n",
      "Collecting sentencepiece\n",
      "  Downloading sentencepiece-0.2.0-cp310-cp310-win_amd64.whl.metadata (8.3 kB)\n",
      "Requirement already satisfied: filelock in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from transformers) (3.13.1)\n",
      "Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from transformers) (0.24.6)\n",
      "Requirement already satisfied: numpy>=1.17 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from transformers) (1.26.4)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from transformers) (24.1)\n",
      "Requirement already satisfied: pyyaml>=5.1 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from transformers) (6.0.1)\n",
      "Requirement already satisfied: regex!=2019.12.17 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from transformers) (2024.7.24)\n",
      "Requirement already satisfied: requests in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from transformers) (2.32.3)\n",
      "Requirement already satisfied: safetensors>=0.4.1 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from transformers) (0.4.4)\n",
      "Requirement already satisfied: tokenizers<0.20,>=0.19 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from transformers) (0.19.1)\n",
      "Requirement already satisfied: tqdm>=4.27 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from transformers) (4.66.5)\n",
      "Requirement already satisfied: typing-extensions>=4.8.0 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from torch) (4.12.2)\n",
      "Requirement already satisfied: sympy in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from torch) (1.12)\n",
      "Requirement already satisfied: networkx in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from torch) (3.3)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from torch) (3.1.4)\n",
      "Requirement already satisfied: fsspec in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from torch) (2024.6.1)\n",
      "Requirement already satisfied: colorama in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from tqdm>=4.27->transformers) (0.4.6)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from jinja2->torch) (2.1.5)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from requests->transformers) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from requests->transformers) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from requests->transformers) (2.2.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from requests->transformers) (2024.7.4)\n",
      "Requirement already satisfied: mpmath>=0.19 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from sympy->torch) (1.3.0)\n",
      "Downloading sentencepiece-0.2.0-cp310-cp310-win_amd64.whl (991 kB)\n",
      "   ---------------------------------------- 0.0/991.5 kB ? eta -:--:--\n",
      "   ---------- ----------------------------- 262.1/991.5 kB ? eta -:--:--\n",
      "   ---------------------------------------- 991.5/991.5 kB 2.7 MB/s eta 0:00:00\n",
      "Installing collected packages: sentencepiece\n",
      "Successfully installed sentencepiece-0.2.0\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install transformers torch sentencepiece"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "671fd8f5-de9a-46a9-b744-c9891ab5606c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Load your training data from the CSV file\n",
    "data_path = 'D:/DATA_VentureX/LLM_Content_generation_data/Youtube_data.csv'\n",
    "df = pd.read_csv(data_path)\n",
    "\n",
    "# Use the 'transcription' column for training\n",
    "transcriptions = df['transcription'].tolist()\n",
    "\n",
    "# Preparing the data in a format suitable for fine-tuning\n",
    "# Here we assume that the model will generate content based on the transcription\n",
    "training_data = [{\"input\": transcription, \"output\": transcription} for transcription in transcriptions]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "5ccb163a-7dce-427e-bdc8-2154653ed953",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages\\transformers\\tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c252627b16cb40dbacf47ed7c682db70",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/924 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='1386' max='1386' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [1386/1386 23:13:19, Epoch 3/3]\n",
       "    </div>\n",
       "    <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       " <tr style=\"text-align: left;\">\n",
       "      <th>Step</th>\n",
       "      <th>Training Loss</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>500</td>\n",
       "      <td>2.029400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1000</td>\n",
       "      <td>1.894000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table><p>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "TrainOutput(global_step=1386, training_loss=1.9036852904040404, metrics={'train_runtime': 83625.6209, 'train_samples_per_second': 0.033, 'train_steps_per_second': 0.017, 'total_flos': 724301512704000.0, 'train_loss': 1.9036852904040404, 'epoch': 3.0})"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments\n",
    "from datasets import Dataset\n",
    "import pandas as pd\n",
    "\n",
    "# Load the tokenizer and model\n",
    "tokenizer = GPT2Tokenizer.from_pretrained('gpt2')\n",
    "model = GPT2LMHeadModel.from_pretrained('gpt2')\n",
    "\n",
    "# Add a padding token to the tokenizer\n",
    "tokenizer.pad_token = tokenizer.eos_token\n",
    "\n",
    "# Load your training data from the CSV file\n",
    "csv_file_path = 'D:/DATA_VentureX/LLM_Content_generation_data/Youtube_data.csv'\n",
    "df = pd.read_csv(csv_file_path)\n",
    "\n",
    "# Ensure the 'transcription' column exists and contains text\n",
    "if 'transcription' not in df.columns:\n",
    "    raise ValueError(\"The 'transcription' column is not found in the CSV file.\")\n",
    "\n",
    "# Prepare training data for the tokenizer\n",
    "training_data = {\n",
    "    'input': df['transcription'].astype(str).tolist()  # Ensure all data is string type\n",
    "}\n",
    "\n",
    "# Prepare the dataset\n",
    "dataset = Dataset.from_pandas(pd.DataFrame(training_data))\n",
    "\n",
    "# Tokenize the data\n",
    "def tokenize_function(examples):\n",
    "    # Check the type of examples['input']\n",
    "    if not isinstance(examples['input'], list):\n",
    "        raise ValueError(\"The input should be a list of strings.\")\n",
    "    \n",
    "    # Tokenize the inputs\n",
    "    encodings = tokenizer(examples['input'], truncation=True, padding='max_length', max_length=512)\n",
    "    \n",
    "    # Add labels as the same as input_ids\n",
    "    encodings['labels'] = encodings['input_ids']\n",
    "    \n",
    "    return encodings\n",
    "\n",
    "# Apply the tokenization\n",
    "tokenized_datasets = dataset.map(tokenize_function, batched=True)\n",
    "\n",
    "# Define the training arguments\n",
    "training_args = TrainingArguments(\n",
    "    output_dir='./results',\n",
    "    per_device_train_batch_size=2,\n",
    "    num_train_epochs=3,\n",
    "    save_steps=10_000,\n",
    "    save_total_limit=2,\n",
    ")\n",
    "\n",
    "# Initialize the Trainer\n",
    "trainer = Trainer(\n",
    "    model=model,\n",
    "    args=training_args,\n",
    "    train_dataset=tokenized_datasets,\n",
    ")\n",
    "\n",
    "# Fine-tune the model\n",
    "trainer.train()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "839d1c7b-04db-44dc-a131-2733af9cbcbf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "('D:/DATA_VentureX/LLM_Content_generation_data/Model_Save/tokenizer_config.json',\n",
       " 'D:/DATA_VentureX/LLM_Content_generation_data/Model_Save/special_tokens_map.json',\n",
       " 'D:/DATA_VentureX/LLM_Content_generation_data/Model_Save/vocab.json',\n",
       " 'D:/DATA_VentureX/LLM_Content_generation_data/Model_Save/merges.txt',\n",
       " 'D:/DATA_VentureX/LLM_Content_generation_data/Model_Save/added_tokens.json')"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "output_dir = 'D:/DATA_VentureX/LLM_Content_generation_data/Model_Save/'\n",
    "\n",
    "trainer.save_model(output_dir)\n",
    "tokenizer.save_pretrained(output_dir)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "634601b6-863e-4fee-935e-429db444be46",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting streamlit\n",
      "  Downloading streamlit-1.37.1-py2.py3-none-any.whl.metadata (8.5 kB)\n",
      "Requirement already satisfied: transformers in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (4.44.2)\n",
      "Collecting altair<6,>=4.0 (from streamlit)\n",
      "  Downloading altair-5.4.1-py3-none-any.whl.metadata (9.4 kB)\n",
      "Collecting blinker<2,>=1.0.0 (from streamlit)\n",
      "  Using cached blinker-1.8.2-py3-none-any.whl.metadata (1.6 kB)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from streamlit) (5.4.0)\n",
      "Requirement already satisfied: click<9,>=7.0 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from streamlit) (8.1.7)\n",
      "Requirement already satisfied: numpy<3,>=1.20 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from streamlit) (1.26.4)\n",
      "Requirement already satisfied: packaging<25,>=20 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from streamlit) (24.1)\n",
      "Requirement already satisfied: pandas<3,>=1.3.0 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from streamlit) (2.2.2)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from streamlit) (10.4.0)\n",
      "Requirement already satisfied: protobuf<6,>=3.20 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from streamlit) (5.27.2)\n",
      "Requirement already satisfied: pyarrow>=7.0 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from streamlit) (17.0.0)\n",
      "Requirement already satisfied: requests<3,>=2.27 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from streamlit) (2.32.3)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from streamlit) (13.7.1)\n",
      "Collecting tenacity<9,>=8.1.0 (from streamlit)\n",
      "  Using cached tenacity-8.5.0-py3-none-any.whl.metadata (1.2 kB)\n",
      "Collecting toml<2,>=0.10.1 (from streamlit)\n",
      "  Downloading toml-0.10.2-py2.py3-none-any.whl.metadata (7.1 kB)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from streamlit) (4.12.2)\n",
      "Collecting gitpython!=3.1.19,<4,>=3.0.7 (from streamlit)\n",
      "  Downloading GitPython-3.1.43-py3-none-any.whl.metadata (13 kB)\n",
      "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
      "  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from streamlit) (6.4.1)\n",
      "Collecting watchdog<5,>=2.1.5 (from streamlit)\n",
      "  Downloading watchdog-4.0.2-py3-none-win_amd64.whl.metadata (38 kB)\n",
      "Requirement already satisfied: filelock in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from transformers) (3.13.1)\n",
      "Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from transformers) (0.24.6)\n",
      "Requirement already satisfied: pyyaml>=5.1 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from transformers) (6.0.1)\n",
      "Requirement already satisfied: regex!=2019.12.17 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from transformers) (2024.7.24)\n",
      "Requirement already satisfied: safetensors>=0.4.1 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from transformers) (0.4.4)\n",
      "Requirement already satisfied: tokenizers<0.20,>=0.19 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from transformers) (0.19.1)\n",
      "Requirement already satisfied: tqdm>=4.27 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from transformers) (4.66.5)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
      "Collecting narwhals>=1.5.2 (from altair<6,>=4.0->streamlit)\n",
      "  Downloading narwhals-1.5.5-py3-none-any.whl.metadata (5.8 kB)\n",
      "Requirement already satisfied: colorama in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from click<9,>=7.0->streamlit) (0.4.6)\n",
      "Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
      "  Downloading gitdb-4.0.11-py3-none-any.whl.metadata (1.2 kB)\n",
      "Requirement already satisfied: fsspec>=2023.5.0 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (2024.6.1)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from pandas<3,>=1.3.0->streamlit) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2.2.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2024.7.4)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from rich<14,>=10.14.0->streamlit) (2.18.0)\n",
      "Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
      "  Downloading smmap-5.0.1-py3-none-any.whl.metadata (4.3 kB)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.5)\n",
      "Requirement already satisfied: attrs>=22.2.0 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.2.0)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.12.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.19.1)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\msarv\\anaconda3\\envs\\ds_env\\lib\\site-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\msarv\\appdata\\roaming\\python\\python310\\site-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)\n",
      "Downloading streamlit-1.37.1-py2.py3-none-any.whl (8.7 MB)\n",
      "   ---------------------------------------- 0.0/8.7 MB ? eta -:--:--\n",
      "   ---------------------------------------- 0.0/8.7 MB ? eta -:--:--\n",
      "   - -------------------------------------- 0.3/8.7 MB ? eta -:--:--\n",
      "   -- ------------------------------------- 0.5/8.7 MB 1.2 MB/s eta 0:00:07\n",
      "   ---- ----------------------------------- 1.0/8.7 MB 1.4 MB/s eta 0:00:06\n",
      "   ------ --------------------------------- 1.3/8.7 MB 1.4 MB/s eta 0:00:06\n",
      "   ------- -------------------------------- 1.6/8.7 MB 1.5 MB/s eta 0:00:05\n",
      "   --------- ------------------------------ 2.1/8.7 MB 1.6 MB/s eta 0:00:05\n",
      "   ------------ --------------------------- 2.6/8.7 MB 1.7 MB/s eta 0:00:04\n",
      "   -------------- ------------------------- 3.1/8.7 MB 1.8 MB/s eta 0:00:04\n",
      "   ---------------- ----------------------- 3.7/8.7 MB 1.9 MB/s eta 0:00:03\n",
      "   -------------------- ------------------- 4.5/8.7 MB 2.1 MB/s eta 0:00:03\n",
      "   ------------------------ --------------- 5.2/8.7 MB 2.2 MB/s eta 0:00:02\n",
      "   --------------------------- ------------ 6.0/8.7 MB 2.4 MB/s eta 0:00:02\n",
      "   -------------------------------- ------- 7.1/8.7 MB 2.6 MB/s eta 0:00:01\n",
      "   ------------------------------------ --- 7.9/8.7 MB 2.6 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 8.7/8.7 MB 2.7 MB/s eta 0:00:00\n",
      "Downloading altair-5.4.1-py3-none-any.whl (658 kB)\n",
      "   ---------------------------------------- 0.0/658.1 kB ? eta -:--:--\n",
      "   --------------- ------------------------ 262.1/658.1 kB ? eta -:--:--\n",
      "   ---------------------------------------- 658.1/658.1 kB 1.8 MB/s eta 0:00:00\n",
      "Using cached blinker-1.8.2-py3-none-any.whl (9.5 kB)\n",
      "Downloading GitPython-3.1.43-py3-none-any.whl (207 kB)\n",
      "Downloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)\n",
      "   ---------------------------------------- 0.0/6.9 MB ? eta -:--:--\n",
      "   --- ------------------------------------ 0.5/6.9 MB 2.8 MB/s eta 0:00:03\n",
      "   ------- -------------------------------- 1.3/6.9 MB 3.5 MB/s eta 0:00:02\n",
      "   --------------- ------------------------ 2.6/6.9 MB 4.2 MB/s eta 0:00:02\n",
      "   ------------------ --------------------- 3.1/6.9 MB 4.2 MB/s eta 0:00:01\n",
      "   ------------------------ --------------- 4.2/6.9 MB 4.1 MB/s eta 0:00:01\n",
      "   ------------------------------ --------- 5.2/6.9 MB 4.1 MB/s eta 0:00:01\n",
      "   ------------------------------- -------- 5.5/6.9 MB 3.9 MB/s eta 0:00:01\n",
      "   ---------------------------------- ----- 6.0/6.9 MB 3.5 MB/s eta 0:00:01\n",
      "   ------------------------------------ --- 6.3/6.9 MB 3.4 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 6.9/6.9 MB 3.2 MB/s eta 0:00:00\n",
      "Using cached tenacity-8.5.0-py3-none-any.whl (28 kB)\n",
      "Downloading toml-0.10.2-py2.py3-none-any.whl (16 kB)\n",
      "Downloading watchdog-4.0.2-py3-none-win_amd64.whl (82 kB)\n",
      "Downloading gitdb-4.0.11-py3-none-any.whl (62 kB)\n",
      "Downloading narwhals-1.5.5-py3-none-any.whl (152 kB)\n",
      "Downloading smmap-5.0.1-py3-none-any.whl (24 kB)\n",
      "Installing collected packages: watchdog, toml, tenacity, smmap, narwhals, blinker, pydeck, gitdb, gitpython, altair, streamlit\n",
      "Successfully installed altair-5.4.1 blinker-1.8.2 gitdb-4.0.11 gitpython-3.1.43 narwhals-1.5.5 pydeck-0.9.1 smmap-5.0.1 streamlit-1.37.1 tenacity-8.5.0 toml-0.10.2 watchdog-4.0.2\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  WARNING: The script watchmedo.exe is installed in 'C:\\Users\\msarv\\anaconda3\\envs\\ds_env\\Scripts' which is not on PATH.\n",
      "  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.\n",
      "  WARNING: The script streamlit.exe is installed in 'C:\\Users\\msarv\\anaconda3\\envs\\ds_env\\Scripts' which is not on PATH.\n",
      "  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit transformers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "05d5388d-8f94-470d-9c28-1f8c46b16003",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-08-27 10:49:57.143 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\msarv\\AppData\\Roaming\\Python\\Python310\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2024-08-27 10:49:57.147 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer\n",
    "\n",
    "# Load the fine-tuned model and tokenizer\n",
    "output_dir = 'D:/DATA_VentureX/LLM_Content_generation_data/Model_Save/'\n",
    "model = GPT2LMHeadModel.from_pretrained(output_dir)\n",
    "tokenizer = GPT2Tokenizer.from_pretrained(output_dir)\n",
    "generator = pipeline(\"text-generation\", model=model, tokenizer=tokenizer)\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"YouTube Content Generation\")\n",
    "\n",
    "# User input\n",
    "prompt = st.text_area(\"Enter a prompt:\")\n",
    "\n",
    "# Generate content\n",
    "if st.button(\"Generate Content\"):\n",
    "    generated = generator(prompt, max_length=200, num_return_sequences=1)\n",
    "    st.write(generated[0]['generated_text'])\n",
    "\n",
    "st.write(\"Fine-tuned model used: gpt2\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "ds_env",
   "language": "python",
   "name": "ds_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
