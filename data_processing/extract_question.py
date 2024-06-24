class Extract_question_answer:
    def extract_math_gsm8k(obj):
        question = obj['question']
        answer = obj['answer']
        return question, answer

    def extract_math_metamathqa(obj):
        question = obj['query']
        answer = obj['response']
        return question, answer

    def extract_translation_fren(obj):
        question = "Translate French to English: " + obj['translation']['fr']
        answer = obj['translation']['en']
        return question, answer

    def extract_translation_deen(obj):
        question = "Translate German to English: " + obj['translation']['de']
        answer = obj['translation']['en']
        return question, answer
        
    def extract_translation_csen(obj):
        question = "Translate Czech to English: " + obj['translation']['cs']
        answer = obj['translation']['en']
        return question, answer

    def extract_translation_fien(obj):
        question = "Translate Finnish to English: " + obj['translation']['fi']
        answer = obj['translation']['en']
        return question, answer

    def extract_translation_roen(obj):
        question = "Translate Romanian to English: " + obj['translation']['ro']
        answer = obj['translation']['en']
        return question, answer

    def extract_translation_ruen(obj):
        question = "Translate Russian to English: " + obj['translation']['ru']
        answer = obj['translation']['en']
        return question, answer

    def extract_translation_tren(obj): 
        question = "Translate Turkish to English: " + obj['translation']['tr']
        answer = obj['translation']['en']
        return question, answer

    def extract_translation_zhen(obj): 
        question = "Translate Chinese to English: " + obj['translation']['zh']
        answer = obj['translation']['en']
        return question, answer       

    def extract_translation_paracrawl_enes(obj):
        question = "Translate Spanish to English: " + obj['translation']['es']
        answer = obj['translation']['en']
        return question, answer

    def extract_translation_paracrawl_enfr(obj):
        question = "Translate French to English: " + obj['translation']['fr']
        answer = obj['translation']['en']
        return question, answer

    def extract_translation_jparacrawl_japanese(obj):
        question = "Translate Japanese to English: " + obj['japanese']
        answer = obj['english']
        return question, answer

    def extract_qa_nq(obj):
        question = obj['question']['text']
        return question, None

    def extract_qa_tqa(obj):
        question = obj['question']
        return question, None

    def extract_summ_cnndm(obj):
        question = "Summarize: " + obj['article']
        answer = obj['highlights']
        return question, answer

    def extract_summ_samsum(obj):
        question = "Summarize: " + obj['dialogue']
        answer = obj['summary']
        return question, answer

    def extract_summ_xsum(obj):
        question = "Summarize: " + obj['document']
        answer = obj['summary']
        return question, answer

    def extract_senti_imdb(obj):
        question = "Read the following movie review carefully and analyze its sentiment. Based on the review's tone, language, and overall sentiment, decide label if it is positive (1) or negative (0) : " \
                    + obj['text']
        answer = obj['label']
        return question, answer

    def extract_mt(obj):
        category = obj['category']
        question = obj['prompt']
        answer = obj['reference']
        if not answer: 
            # answer is empty
            answer = None
        return category, question, answer