import streamlit as st
from pyaspeller import YandexSpeller
from io import StringIO
import torch
from streamlit_autorefresh import st_autorefresh
from foo import *

# Функция для обработки файла JSON
def process_json(file_contents):
    try:
        data = json.loads(file_contents)
        # В этом примере просто выводим содержимое файла
        return json.dumps(data, indent=4)
    except Exception as e:
        return f"Ошибка обработки файла: {e}"

# Функция для подсчета количества слов в тексте
def count_words(text):
    words = text.split()
    return len(words)

mistake = 0
toxic = ''
destr = ''

def main():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration, AutoModel
    except Exception as e:
        #st.error(f"Ошибка: {e}")
        st_autorefresh(interval=5000)


    st.title("Веб-приложение для анализа статей Знание.Вики")
    event = st.radio(
        "Выберите способ ввода данных о статье:",
        ('JSON-файл', 'URL')
    )
    uploaded_file = None
    url = ''
    # Логика для обработки выбора
    if event == 'JSON-файл':
        uploaded_file = st.file_uploader("Загрузите файл JSON", type=["json"])

    else:
        url = st.text_input(label="Введите URL статьи")


    # Загрузка файла JSON
    if uploaded_file is not None or url != '':
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            st.code(string_data)
            string_data, table = extract_content_and_remove_from_string(string_data)
            outer_links, inner_links, string_data = clean_json(string_data)
            out_links = []
            for string in outer_links:
                start_index = string.find("url=")
                end_index = string.find("|", start_index)

                if start_index != -1 and end_index != -1:
                    substring = string[start_index + len("url="):end_index]
                    out_links.append(substring)

            title, string_data, images, type_text = parse_json_string(string_data)
            string_data = string_data.replace("}", "")
            string_data = string_data.replace("{", "")
            string_data = process_text_only(string_data)   # text


        else:
            json_data = extract_data_from_page(url)
            st.code(json_data)
            title, string_data, images, out_links = parse_json_string_from_url(json_data)
            # Регулярное выражение для поиска подстрок, начинающихся с "url = " и оканчивающихся пробелом
            title = title.split(' - ')[0]                   # title
            type_text = ["znanierussia.ru/articles"]
            liststr = string_data.split('\n')
            string_data = '\n'.join(liststr[4:])
            string_data = string_data.replace('\n', '')
            string_data = string_data.replace('\t', '')
            inner_links = string_data.count("[")                 # inner_links
            pattern = re.compile(r'\[\w+\\]')
        # Заменяем найденные подстроки на пустую строку
            string_data = re.sub(pattern, '', string_data)           # text
            string_data = string_data.replace("}", "")
            string_data = string_data.replace("{", "")



        st.write("Файл успешно загружен!")

        # Выбор функции обработки
        option = st.selectbox("Выберите функцию обработки",
                              ("Просмотр содержимого",
                               "Анализ деструктивного контента",
                               "Выявление ненормативной лексики",
                               "Выявление и исправление ошибок",
                               "Проверка ссылок на источники иноагентов",
                               "Проверка соответствия заголовка содержанию",
                               "Определение соотношения полезного контента",
                                #"Создать полный отчет"
                               ))

        if option == "Создать полный отчет":
            # Пример функции обработки строки
            def process_string(s):
                return s[::-1]  # Пример: переворачиваем строку

            # Список строк
            strings = ["Анализ деструктивного контента",
                       "Выявление ненормативной лексики",
                       "Выявление ошибок",
                       "Ссылоки на источники иноагентов",
                       "Проверка соответствия заголовка содержанию",
                       "Определение соотношения полезного контента"
                       ]

            # Обрабатываем строки
            processed_strings = []  # [process_string(s) for s in strings]

            processed_strings.append(f"{toxic}, {destr}")
            processed_strings.append(0)
            processed_strings.append(f"{mistake} ошибок")
            processed_strings.append(4)
            processed_strings.append(5)
            processed_strings.append(6)

            # Создаем DataFrame
            data = {
                "Проверка": strings,
                "Результат": processed_strings
            }
            import pandas as pd
            df = pd.DataFrame(data)

            # Отображаем таблицу в Streamlit
            st.write("### Таблица результатов обработки")
            st.dataframe(df)

        # Обработка в зависимости от выбора пользователя
        if option == "Просмотр содержимого":
            st.subheader("Содержимое файла:")
            st.write("I. Заголовок статьи:")
            st.code(title)
            st.write("II. Текст статьи:")
            st.code(string_data)
            st.write("III. Список ссылок на внешние источники:")
            selected = st.selectbox("Выберите ссылку:", out_links)

            # Отображение выбранного элемента
            st.write(f"Вы выбрали: {selected}")
            st.write(f"IV. Количество ссылок на статьи Знание.Вики: {inner_links}")
            st.write("V. Список ссылок на изображения:")
            #st.code('\n'.join(images))
            # Отображение выпадающего списка для выбора элемента
            selected_item = st.selectbox("Выберите изображение:", images)

            # Отображение выбранного элемента
            st.write(f"Вы выбрали: {selected_item}")
            st.write(f"VI. {type_text}")



        elif option == 'Проверка соответствия заголовка содержанию':
            article_title = title
            if article_title:
                st.write("Введенное название статьи:", article_title)


            from huggingface_hub import hf_hub_download
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration, \
                AutoModel
            # Загрузка файла .pth из репозитория Hugging Face
            repo_id = "Vlad1m/check_topic"  # замените на ваш репозиторий
            filename = "model.pth"  # замените на имя вашего файла .pth
            file_path = hf_hub_download(repo_id=repo_id, filename=filename)

            # Определение архитектуры модели
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_name = "IlyaGusev/rut5_base_headline_gen_telegram"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name, resume_download=True)
            model.load_state_dict(torch.load(file_path, map_location=device))
            model.to(device)

            def generate_text(input_text, max_length=(len(string_data) // 4)):
                input_ids = tokenizer.encode_plus(
                    input_text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt"
                )["input_ids"].to(device)

                attention_mask = tokenizer.encode_plus(
                    input_text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt"
                )["attention_mask"].to(device)

                output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)
                output_text = tokenizer.decode(output[0], skip_special_tokens=True)
                return output_text

            pred_title = generate_text(string_data)
            pred_title2 = str(pred_title)

            def clean_string(s):
                # Заменяем все, что не является буквой, цифрой или пробелом, на пробел
                s = re.sub(r'[^\w\s]', ' ', s)
                s = re.sub("_", " ", s)
                # Заменяем два и более пробелов на один пробел
                s = re.sub(r'\s+', ' ', s)
                return s

            # Очистка строки
            cleaned_string = clean_string(pred_title2)

            st.write(f"Предполагаемое название статьи: {cleaned_string}")

            # Функция для сравнения текстов

            def compare_texts(text1, text2):

                vectorizer = TfidfVectorizer().fit_transform([text1, text2])

                vectors = vectorizer.toarray()

                cos_sim = cosine_similarity(vectors)

                return cos_sim[0, 1]

            similarity = compare_texts(cleaned_string, article_title)

            if similarity >= 0.6:

                st.markdown('<span style="color:green;">Название соответствует содержанию статьи.</span>',
                            unsafe_allow_html=True)

            else:

                st.markdown('<span style="color:red;">Название не соответствует содержанию статьи.</span>',
                            unsafe_allow_html=True)
        elif option == "Анализ деструктивного контента":
            model = AutoModelForSequenceClassification.from_pretrained('Vlad1m/toxicity_analyzer')
            tokenizer = AutoTokenizer.from_pretrained('Vlad1m/toxicity_analyzer')

            def get_sentiment(text):
                with torch.no_grad():
                    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(
                        model.device)
                    proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]

                return (model.config.id2label[proba.argmax()])  # proba


            st.markdown(':red[Результаты проверки на токсичность:]')
            st.write(get_sentiment(string_data))

            model = AutoModelForSequenceClassification.from_pretrained('Vlad1m/destractive_context')
            tokenizer = AutoTokenizer.from_pretrained('Vlad1m/destractive_context')
            st.markdown(':red[Результаты проверки на деструктивный контент:]')
            tox = get_sentiment(string_data)
            st.write(tox)



        elif option == "Выявление ненормативной лексики":
            # Список слов для выделения красным цветом
            word_list = read_lines_from_file('rudewords.txt')
            highlighted_text = highlight_words_in_text(string_data, word_list, "red")
            word_list = read_lines_from_file('destr.txt')
            highlighted_text = highlight_words_in_text(highlighted_text, word_list, "green")
            st.markdown(highlighted_text, unsafe_allow_html=True)

        elif option == 'Выявление и исправление ошибок':
            speller = YandexSpeller()
            data_correct = speller.spelled(string_data)
            if string_data == data_correct:
                st.write('Нет ошибок')
            else:
                st.markdown(':red[Статья содержит ошибки]')
                mist = len(set(string_data.split()) - set(data_correct.split()))
                st.write(f'Количество исправленных слов: :blue[{mist}]')
                st.write(f'Исправлены слова: :blue[{set(string_data.split()) - set(data_correct.split())}]')
                st.text_area('Исправленный текст:', data_correct, height=300)

            mist_list = read_lines_from_file('Ling.txt')
            highlighted_text = highlight_words_in_text(string_data, mist_list, "red")
            st.markdown(highlighted_text, unsafe_allow_html=True)


        elif option == 'Проверка ссылок на источники иноагентов':
            import pandas as pd
            progress_bar = st.progress(0)
            from natasha import (
                Segmenter,
                NewsEmbedding,
                NewsMorphTagger,
                NewsSyntaxParser,
                Doc,
                NewsNERTagger,
                MorphVocab
            )
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            from joblib import Parallel, delayed

            # Инициализация компонентов Natasha
            segmenter = Segmenter()
            emb = NewsEmbedding()
            morph_tagger = NewsMorphTagger(emb)
            syntax_parser = NewsSyntaxParser(emb)
            ner_tagger = NewsNERTagger(emb)
            morph_vocab = MorphVocab()

            # Обработка текста с Natasha
            doc = Doc(string_data)
            doc.segment(segmenter)
            doc.tag_morph(morph_tagger)
            doc.parse_syntax(syntax_parser)
            if doc.sents:
                sent = doc.sents[0]
                # Можно продолжить работу с первым предложением
            else:
                # Обработка случая, когда предложений не найдено
                # Например, можно вывести сообщение об ошибке или обработать исключение иначе
                print("В тексте не найдено ни одного предложения.")
            doc.tag_ner(ner_tagger)

            for token in doc.tokens:
                token.lemmatize(morph_vocab)

            for span in doc.spans:
                span.normalize(morph_vocab)

            vse_imena_v_text = {_.text: _.normal for _ in doc.spans}
            df3 = pd.DataFrame(list(vse_imena_v_text.items()), columns=['name', 'clean_name'])

            # Очистка текста
            df3['clean_text'] = df3['name'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x)).apply(
                lambda x: re.sub(r'\s+', ' ', x).strip())

            df4 = pd.read_csv('inoagenty.csv', sep='\t')
            df4['clean_text'] = df4['name'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x)).apply(
                lambda x: re.sub(r'\s+', ' ', x).strip())

            def compare_texts(index1, index2):
                text1 = "".join(df3['clean_name'][index1])
                text2 = "".join(df4['clean_text'][index2])
                vectorizer = TfidfVectorizer().fit_transform([text1, text2])
                vectors = vectorizer.toarray()
                cos_sim = cosine_similarity(vectors)
                return cos_sim[0, 1], text1

            def process_one_row(i):
                local_black = []
                for j in range(len(df3)):
                    similarity, name = compare_texts(j, i)
                    if similarity >= 0.5799:
                        local_black.append(name)
                return local_black

            results = []
            black = []
            for i in range(len(df4)):
                local_black = process_one_row(i)
                black.extend(local_black)
                # Обновляем прогресс-бар после каждой итерации
                progress_bar.progress((i + 1) / len(df4))

            # black = [name for sublist in results for name in sublist]

            counter = len(black)
            st.write(f"Количество найденных отсылок на иноагентов: {counter}")

            i = 1
            for one in black:
                st.write(f"{i}. {one}")
                i += 1

            def highlight_words(text, phrases_to_highlight):
                phrases_to_highlight.sort(key=len, reverse=True)
                for phrase in phrases_to_highlight:
                    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                    text = pattern.sub(f"<span style='color:red;'>{phrase}</span>", text)
                return text

            black_phrases = [phrase for sublist in black for phrase in sublist.split()]

            highlighted_text = highlight_words(string_data, black_phrases)

            st.markdown(highlighted_text, unsafe_allow_html=True)



        elif option == 'Определение соотношения полезного контента':
            st.subheader("Материалы данной статьи:")
            st.write(f"Упомянуто ссылок на статьи znanierussia.ru: {inner_links}")
            st.write(f"Упомянуто ссылок на сторонние ресурсы: {len(out_links)}")
            st.write(f"Использовано изображений в статье: {len(images)}")

            def article_optimization(text, num_images, num_links):
                # Определяем длину текста статьи
                article_length = len(text.split(" "))

                # Рассчитываем оптимальное количество ссылок
                optimal_links = min(article_length // 300, num_links)
                link_ratio = optimal_links / article_length

                # Рассчитываем оптимальное количество изображений
                optimal_images = min(article_length // 500, num_images)
                image_ratio = optimal_images / article_length

                return link_ratio, image_ratio

            link_ratio, image_ratio = (article_optimization(string_data, len(images), len(out_links)))

            if link_ratio < 0.0033:  # 1 ссылка на каждые 300 слов
                st.markdown(f"Количество ссылок <span style=\"color:blue\">ниже</span> оптимального.", unsafe_allow_html=True)
            else:
                st.markdown("Количество ссылок оптимально или <span style=\"color:red\">выше</span> оптимального.", unsafe_allow_html=True)

            if image_ratio < 0.002:  # 1 изображение на каждые 500 слов
                st.markdown("Количество изображений <span style=\"color:blue\">ниже</span> оптимального.", unsafe_allow_html=True)
            else:
                st.markdown("Количество изображений оптимально или <span style=\"color:red\">выше</span> оптимального.", unsafe_allow_html=True)



    else:
        st.write("Загрузите файл, чтобы начать работу")

if __name__ == "__main__":
    main()