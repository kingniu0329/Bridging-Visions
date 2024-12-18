import spacy
from word2number import w2n
import inflect
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import requests
# text
def gettext_result(sentence):
    nlp = spacy.load("en_core_web_sm")
    p = inflect.engine()
    def plural_to_singular(word):
        # 使用 inflect 库将复数转换为单数
        return p.singular_noun(word) or word  #

    def keep_quantifiers_and_nouns(sentence):
        # Process the sentence using spaCy
        doc = nlp(sentence)
        res = []
        for token in doc:
            if token.pos_ == 'NUM':
                res.append(token.text)
            elif token.pos_ == 'NOUN':
                res.append(plural_to_singular(token.text))
        # Keep only quantifiers (like numbers) and nouns

        # Join the result into a sentence
        return ' '.join(res)

    def word_to_number(word):
        try:
            # 将单词转换为数字
            return w2n.word_to_num(word)
        except ValueError:
            # 处理无法转换的情况
            return None
    def getdirt(sentence):
        doc = nlp(sentence)
        result = {}
        num = 1
        for token in doc:
            if token.pos_ == 'NUM':
                num = word_to_number(token.text)
            elif token.pos_ == 'NOUN':
                result[token.text] = num
                num = 1
        return result

    # Example sentence
    text_result = getdirt(keep_quantifiers_and_nouns(sentence))
    return text_result

# iamge
def getimage_result(image_path):
    image = Image.open(image_path)
    model = YolosForObjectDetection.from_pretrained('../../model/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained("../../model/yolos-tiny")

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    logits = outputs.logits
    bboxes = outputs.pred_boxes


    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    image_result = {}
    image_confidence = {}
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score.item() > 0.8:
            if model.config.id2label[label.item()] not in image_result:
                image_result[model.config.id2label[label.item()]] = 1
            else:
                image_result[model.config.id2label[label.item()]] += 1
            if model.config.id2label[label.item()] not in image_confidence:
                image_confidence[model.config.id2label[label.item()]] = score.item()
            else :
                image_confidence[model.config.id2label[label.item()]] += score.item()
    return image_result, image_confidence
# compare
def getF1score(imagePath, prompt):
    text_result = gettext_result(prompt)
    image_result, image_confidence = getimage_result(imagePath)
    print(image_result)
    print(image_confidence)
    print(text_result)
    N = len(text_result)
    print(N)
    Macro_Precision = 0
    Macro_Recall = 0
    for key in text_result:
        if key in image_result:
            Macro_Precision += image_confidence[key]/image_result[key]
            Macro_Recall += min(image_result[key],text_result[key])/max(image_result[key],text_result[key])
    if N == 0:
        return 0
    Macro_Precision = Macro_Precision/N
    Macro_Recall = Macro_Recall/N
    if Macro_Precision+Macro_Recall == 0:
        f1 = 0
    else:
        f1 = 2*Macro_Precision*Macro_Recall/(Macro_Precision+Macro_Recall)
    return f1

# test
imagePath = "samples_images copy/image_78.png"
prompt = " three skiers look over edge of balcony to the slopes below."
print(getF1score(imagePath, prompt))