# OCR справок о донорстве крови

## Цель работы
Найти готовое решение или разработать своё, которое позволит находить и считывать данные из таблицы в файл в медицинской форме № 405-05/у о донорстве крови.

## Решаемые задачи
Для выполнения проекта мы поставили перед собой следующие задачи.
1. Изучить существующие решения в области OCR таблиц.
2. Выбрать несколько решений и протестировать их.
3. Среди протестированных решений выбрать лучшее и доработать его.
4. Проверить работу подготовленного решения на данных подготовленных заказчиком и рассчитать итоговую метрику качества accuracy*.
5. Разработать простой интерфейс для взаимодействия с пользователем.
6. Контейнеризировать решение с помощью Doker.

\* В данном проекте метрика accuracy для одной таблицы рассчитывалась как процент ячеек, текст в которых был распознан правильно в соответствии с разметкой данных, которую предоставил заказчик. Итоговое значение метрики рассчитывалось как среднее значение по всем изображениям медицинских форм, предоставленных заказчиком.

Пример формы № 405-05/у и того, как должна выглядеть итоговая таблица, составленная на её основании.

<img src="https://github.com/olga-khrushcheva/DonorSearch/blob/master/images/Example_form_405-05.jpg" height=50% width=50%>

<img src="https://github.com/olga-khrushcheva/DonorSearch/blob/master/images/Example_of_recognized_table.png">

## Разработанное решение
За основу решения был взят класс **TableExtractor** из [этого репозитория](https://github.com/livefiredev/ocr-extract-table-from-image-python), который мы доработали, добавив новые функции. Данный класс позволяет находить и извлекать таблицы из изображений с помощью методов библиотеки **OpenCV**, а также производить их предобработку. Извлечённое изображение таблицы затем обрабатывается с помощью методов библиотеки **img2table** с поддержкой инструмента OCR **EasyOCR**; в результате на выходе получается сформированная по изображению таблица, которую пользователь может посмотреть и скачать.

Итоговое решение выглядит следующим образом.
1.  Извлечение изображения таблицы с помощью класса **TableExtractor**.

    1.1. Предобработка изображения, которую тоже условно можно разбить на несколько последовательных этапов.
    * Первичная обрезка исходного изображения на 40% сверху, поскольку, как заметно на изображении выше, таблица располагается только на второй половине листа с формой № 405-05/у. Затем выполняются нормализация изображения, удаление печатей на нём, удаление шума, повышения яркости, контрастности и чёткости изображения.
    * С помощью модели [detr-doc-table-detection](https://huggingface.co/TahaDouaji/detr-doc-table-detection) находится область, в которой располагается таблица. Данная модель не учитывает возможный наклон таблиц на изображениях, поэтому чтобы не обрезать часть таблицы, от координат ограничивающего таблицу прямоугольника, найденного этой моделью, делается небольшой отступ. По этим координатам вырезаем область, в которой расположена таблица.
    * Выполнение оставшихся этапов предобработки изображения перед более точным извлечением таблицы, а именно: бинаризация изображения, его инверсия и утолщение контуров с помощью функции **cv2.dilate()** с параметром **iterations=2**, чтобы точнее определить контуры таблицы на изображении.
      
    1.2. Поиск всех контуров на изображении, и выбор только прямоугольных, среди оставшихся выбирается наибольший по площади контур.

    1.3. Если найденный на предыдущем этапе прямоугольник имеет площадь меньшую, чем *0.06 * высота исходного изображения * 0.5 * ширины исходного изображения*, то ещё раз выполняется утолщение контуров изображения с помощью функции **cv2.dilate()**, только теперь параметр **iterations=1**, а затем повторно выполняется предыдущий пункт 1.3. Дело в том, утолщение контуров с помощью функции **cv2.dilate()** с параметром **iterations=2** затем позволяло достаточно точно находить контур таблицы на большинстве изображений, однако, на некоторых изображениях некоторые контуры начинали сливаться друг с другом, что нарушало контур искомой таблицы в результате чего таблица не находилась. Поэтому к таким изображениям необходимо применять функцию **cv2.dilate()** с параметром **iterations=1**.

    1.4. Иcправление перспективы изображения: если таблица на изображении как-либо наклонена, то на этом этапе происходит её выравнивание. При исправлении перспективы мы также берём небольшой отступ от угловых точек прямоугольного контура таблицы, чтобы не обрезать её края. Стоит отменить, что если таблица смята, то данное преобразование это не исправит.

    Результат работы класса **TableExtractor**:
    <img src="https://github.com/olga-khrushcheva/DonorSearch/assets/137498906/97eb5355-b281-4042-96c1-73cc1e258b2e">


2. После работы класса **TableExtractor**, вырезанная с фото таблица при помощи **img2table** переводится в формат exel ([img2table](https://github.com/xavctn/img2table) - библиотека, предназначенная для распознавания и извлечения таблиц). В данной библиотеке реализована возможность использования различных OCR для распознавания текста в таблице. В данной работе были протестированы три различных варианта OCR: Tesseract, Paddle OCR, EasyOCR. Наилучших результатов удалось добиться с помощью **EasyOCR**.

   Результат работы img2table с использованием EasyOCR:
    <img src="https://github.com/olga-khrushcheva/DonorSearch/assets/137498906/708735b6-c337-42d7-bd96-86b65f128efa">

3. Далее полученная exel таблица обрабатывется с помощью класса **TableToCsv**, который переводит таблицу exel в csv формат, необходимый заказчику. Это и есть конечный результат работы программы.

   Результат работы класса **TableToCsv**:
    <img src="https://github.com/olga-khrushcheva/DonorSearch/assets/137498906/9cc1adb9-3034-42ee-baae-dec5dfd0b402">

4. Для проверки качества работы программы была использована метрика accuracy, итоговое значение которой на тестовом датасете около **53%**. Дело в том, что значение метрики и итоговый результат напрямую зависят от качества изображений, попадающий в программу, поэтому чем качественнее изображение (отсутствие изгибов, равномерное освещение, не мятое фото), тем больше вероятность получить хороший результат.

5. Для более простого и удобного взаимодействия пользователя с программой, было реализовано веб-приложение с использованием Python-фреймворка Streamlit.

    5.1 Пользователь заходит и сайт, где у него есть возможность загркзить изображение в формате jpg:
   
    <img src="https://github.com/olga-khrushcheva/DonorSearch/assets/137498906/3e40b0b6-994f-48b3-adf0-f56f03b166cb)">

    5.2 Далее происходит работа программы, в результате работы которой появляется возможность скачать итоговый csv файл и промежуточную exel таблицу (для более простого форматирования).


