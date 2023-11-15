from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import BertTokenizer
from rest_framework.decorators import api_view

from transformers import BertModel

import torch
from torch import nn

import json
import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pre_trained_model_ckpt = 'cointegrated/rubert-tiny2'

class BertClass(torch.nn.Module):
    def __init__(self, n_classes):
        super(BertClass, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_ckpt,return_dict=False)
        self.drop = torch.nn.Dropout(p = 0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids = input_ids,
            attention_mask= attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

@csrf_exempt
@api_view(['POST'])
def classify_text(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        text = data.get('text', '')

        # Загрузка BERT модели и токенизатора
        tokenizer = BertTokenizer.from_pretrained(pre_trained_model_ckpt)
        class_names = ['positive', 'negative']
        myModel = BertClass(len(class_names))
        myModel.load_state_dict(torch.load('torchapp/model/best_model_state.bin', map_location=device))
        myModel = myModel.to(device)

        encoded_review = tokenizer.encode_plus(text, max_length=512, add_special_tokens=True, return_token_type_ids=False, pad_to_max_length=True, return_attention_mask=True, truncation=True, return_tensors='pt')
        input_ids = encoded_review['input_ids'].to(device)
        attention_mask = encoded_review['attention_mask'].to(device)
        output = myModel(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)

        # Формирование ответа
        response_data = {
            'text': text,
            'probabilities': prediction.tolist()
        }

        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid request method'})
