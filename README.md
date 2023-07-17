# OCR справок о донорстве крови

## Цель работы
Найти готовое решение или разработать своё, которое позволит находить и считывать данные из таблицы в файл в медицинской форме № 405-05/у о донорстве крови.

## Решаемые задачи
Для выполнения проекта мы поставили перед собой следующие задачи.
1. Изучить существующие решения в области OCR таблиц.
2. Выбрать несколько решений и протестировать их.
3. Среди протестированных решений выбрать лучшее и доработать его.
4. Проверить работу подготовленного решения на данных подготовленных заказчиком и рассчитать итоговую метрику качества accuracy*.
5. Контейнеризировать решение с помощью Doker.
6. Разработать простой интерфейс для взаимодействия с пользователем.

\* В данном проекте метрика accuracy для одной таблицы рассчитывалась как процент ячеек, текст в которых был распознан правильно в соотвествии с разметкой данных, которую предоставил заказчик. Итоговое значение метрики расчитывалось как среднее значение по всем изображениям медицинских форм, предоставленных заказчиком.

Пример формы и того, как должна выглядеть итоговая таблица, составленная на её основании.

<img src="https://github.com/olga-khrushcheva/DonorSearch/blob/master/images/Example_form_405-05.jpg" height=40% width=40%>

<img src="https://github.com/olga-khrushcheva/DonorSearch/blob/master/images/Example_of_recognized_table.png">

## Разработанное решение
За основу решения был взят класс **TableExtractor** из [этого репозитория](https://github.com/livefiredev/ocr-extract-table-from-image-python), который мы доработали, добавив новые функции. Данный класс позволяет находить и извлекать таблицы из изображений с помощью методов библиотеки **OpenCV**, а также производить их предобработку. Извлечённое изображение таблицы затем обрабатывается библиотеков **img2table** с поддержкой инструмента OCR **EasyOCR**; в результате на выходе получается сформированная по изображению таблица, которую пользователь может посмотреть и скачать.

Итоговое решение выглядит следующим образом.
1.  Извлечение изображения таблицы с помощью класса **TableExtractor**.
    1.1. Предобработка изображения, которую тоже условно можно разбить на несколько последовательных этапов.
    * Сначала происходит перввичная обрезка исходного изображения на 40% сверху, поскольку, как заментно на изображении выше, таблица располагается только на второй половине листа с формой № 405-05/у. Затем выполняются нормализация изображения, удаление печатей на нём, удаление шумаЮ повышения яркости, конрастности и чёткости изображения.
    * Далее с помощью модели [detr-doc-table-detection](https://huggingface.co/TahaDouaji/detr-doc-table-detection) находится облась, в которой располагается таблица. Данная модель не учитывает возможный наклон таблиц на изображениях, поэтому чтобы не обрезать часть таблицы, от координат ограничивающего таблицу прямоугольника, найденного этой моделью
