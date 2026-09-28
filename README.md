![Needle](assets/banner.svg)

Базовая модель для мобильных устройств, носимой электроники, роботов, умного дома, автомобилей и микроконтроллеров. Вся модель — это один двоичный файл размером 8–29 МБ, построенный на нашей Simple Attention Network; мы жертвуем общей способностью к чату, чтобы обходить модели в 10 раз крупнее на мобильных вызовах инструментов и догонять модели в 2–3 раза крупнее на извлечении данных.

- **Вызовы инструментов**: зная функции, которые публикует ваше приложение, Needle выбирает нужные и заполняет все аргументы из слов пользователя. Попросите две вещи — получите два вызова по порядку; попросите то, что не покрывается ни одним инструментом, — получите пустой список, а не догадку.
- **Структурированное извлечение**: объявите форму, передайте неструктурированный текст — и получите типизированные поля: счёт, бронирование, уведомление, форму. Грамматика декодирования гарантирует, что вывод разбирается корректно, а извлечение обобщается до классификации.
- **Векторные представления текста**: та же модель возвращает вектор для предложения, поэтому приложение может искать, сопоставлять и маршрутизировать локально.

![Needle 3 с одного взгляда](assets/model.svg)

Needle 3 — это «лестничная» Simple Attention Network: Monarch Hadamard MLP вместо FFN, GQA-внимание с причинными свёрточными отводами, n-gram память engram, считываемая через gather, и многополосные гипер-связи; модель обучена так, что каждая глубина от 2 до 20 слоёв является развёртываемой моделью. Большая часть параметров находится в engram, поэтому модель на 121M выполняет арифметику модели на 50M. Байтовая грамматика, скомпилированная из ваших схем, ограничивает каждый токен, а каждый ответ несёт откалиброванную оценку уверенности от обученной головы. Схема архитектуры — на [странице релиза](https://cactuscompute.com/needle).

## Бенчмарки

Для вызова инструментов — точность exact-match на полных тестовых выборках, для извлечения — полевая micro-F1 на полных тестовых выборках.

![Needle 3 против базовых моделей на шести бенчмарках](assets/benchmarks.svg)

Интерактивный график фронтира, архитектура и результаты тонкой настройки — на [cactuscompute.com/needle](https://cactuscompute.com/needle).

## Начните работу

```sh
pip install cactus-needle
```

Попробуйте в браузере на [cactuscompute.com/needle](https://cactuscompute.com/needle); веса и движки для всех платформ находятся на [Hugging Face](https://huggingface.co/Cactus-Compute/needle3).

Задекорируйте функцию: сигнатура задаёт типы аргументов, docstring — описание инструмента, а `run()` замыкает цикл, исполняя вашу функцию и возвращая её результаты.

```python
import needle

@needle.tool
def get_weather(city: str):
    "Get the current weather for a city."
    return {"city": city, "temp_c": 27, "sky": "clear"}

agent = needle.Needle(tools=[get_weather])
print(agent.run("what's it like in Lagos right now?")["results"])
# [{'city': 'Lagos', 'temp_c': 27, 'sky': 'clear'}]
```

Каждый ход возвращает один JSON-объект с `function_calls`, `reasoning` модели и откалиброванной `confidence`; запрос не по теме возвращает пустой список, а не догадку. `needle.Needle(tools=[...], generation=2)` продолжает использовать Needle 2 для существующих развёртываний.

## Руководства

- [Как проектировать инструменты для Needle 3](https://cactuscompute.com/blog/designing-tools-for-needle): один инструмент на действие, названия, как их скажет пользователь, форматы в описаниях, ограничения в грамматике, триггеры.
- [Использование уверенности Needle](https://cactuscompute.com/blog/needle-confidence): что измеряет оценка, что движок скрывает, и маршрутизация на действие, подтверждение или отказ.
- [Структурированное JSON-извлечение с Needle](https://cactuscompute.com/blog/structured-extraction-with-needle): запись как единственный инструмент, типизированные результаты, классификация через enum.
- [Тонкая настройка Needle](https://cactuscompute.com/blog/finetuning-needle): формат данных, команды, чтение лосса, размер датасета.
- [Документация Needle на Python](https://cactuscompute.com/blog/needle-python-docs): API, форма ответа, контракт поведения, системные факты, поиск инструментов, офлайн-устройства, окружения, CLI.
- [Какие устройства поддерживаются в Needle](https://cactuscompute.com/blog/needle-supported-devices): каждая папка платформы, раннер CLI, C API, браузер, WASI, установка без сети.
- [Формат .cact](https://cactuscompute.com/blog/cact-format): файл, который движок отображает на память и читает на месте, Cactus Quants при 2,125 битах на вес, и как разобрать его самостоятельно.
- [Портирование Needle 3](https://cactuscompute.com/blog/porting-needle): заметки для написания собственного рантайма, оракул для проверки, порядок тензоров, который гарантирует контейнер, промпт «на проводе», правило лестницы, поиск через `needle_embed`.

Файл `llms.txt` в этом репозитории содержит ту же справочную информацию для AI-ассистентов по коду.

## Кастомизация

Needle спроектирована так, чтобы её настраивали под себя. Её ёмкость — это лестница, и подобная сеть из всего 2 слоёв, тонко настроенная на инструментах одного продукта, оптимально работает на устройствах, которые намного меньше, чем требует полная модель. Тонкая настройка на DroidCall поднимает каждую подсеть на 18–36 пунктов, а начиная с 4 слоёв настроенная подсеть обходит DeepSeek V4 Flash, начиная с 29M параметров.

![Каждая подсеть до и после тонкой настройки на DroidCall и Mobile Actions](assets/finetune.svg)

Два способа тонкой настройки — из одного и того же пакета:

| | Локально, `needle finetune` | Платформа, `needle platform finetune` |
| --- | --- | --- |
| Что обучается | LoRA-адаптеры на проекциях внимания, база заморожена, сливается при экспорте | Полная модель, каждая глубина начиная с 2 слоёв |
| Что сохраняется | Только ваши данные | Ваши данные, подкреплённые исходным датасетом Needle, чтобы ранее изученное не забывалось |
| Уверенность | Головка не трогается; `confidence` равна `None` | Головка дообучается вместе с моделью, калибруется на ваших инструментах |
| Точность | 4 бита | 2 бита, такое же пост-обучение, как у поставляемой модели |
| Данные | Ваш JSONL, формат `query`/`answers` или чат-формат | Ваши или сгенерированные из ваших определений инструментов, от 100 до 10 000 примеров за запуск |
| Метрики | Валидационный лосс | Точность на валидации и тесте для каждой глубины |
| Вычисления | Ваша машина, JAX на CPU, CUDA или Metal | GPU Cactus |
| Запускается из | CLI | CLI, Python, [дашборда](https://cactuscompute.com/dashboard) или кодового агента с вашим ключом |

Локально:

```sh
pip install "cactus-needle[train]"
needle finetune data.jsonl --epochs 10 --out adapter.safetensors
needle build --lora adapter.safetensors --layers 8 --out tuned.cact
```

Платформа — с ключом из [консоли](https://cactuscompute.com/dashboard/api-keys) в `NEEDLE_API_KEY`. Одна команда загружает файлы, обучает и оценивает каждый размер и скачивает `.cact`-файлы; после отправки задания за ним можно также следить на дашборде:

```sh
export NEEDLE_API_KEY=needle_ft_...
needle platform generate --tools tools.json --examples 1000 --out ./data
needle platform finetune data/train.jsonl data/validation.jsonl data/test.jsonl --suffix smart-home --out ./models
```

```python
from needle.platform import Platform

client = Platform()
job = client.wait(client.finetune(["train.jsonl"], ["validation.jsonl"], ["test.jsonl"], suffix="smart-home"))
paths = client.download(job["fine_tuned_model"], "models", depth=8)
```

Или передайте ключ Claude Code или Codex вместе с [cactuscompute.com/llms.txt](https://cactuscompute.com/llms.txt) и позвольте агенту выполнить весь цикл. `needle platform jobs | models | files | billing` показывают, что хранится в аккаунте, `needle download model-<id>` загружает модель по id, а [руководство по тонкой настройке](https://cactuscompute.com/blog/finetuning-needle) описывает формат данных и чтение метрик.

## Развёртывание

Каждая цель развёртывания поставляет предсобранный движок менее 1 МБ, который при старте загружает веса `needle3.cact`. `needle build --platform <папка> [--layers N]` загружает этот движок и кладёт веса рядом с ним.

![Один движок на папку платформы](assets/deploy.svg)

```sh
needle build --platform macos-arm64
needle build --platform linux-arm64 --layers 8 --out ./pi
./macos-arm64/needle --model needle3.cact --tools tools.json --serve
```

[Руководство по устройствам](https://cactuscompute.com/blog/needle-supported-devices) перечисляет все папки и их содержимое.

По умолчанию телеметрия в двоичном файле включена. Чтобы выключить её, установите переменные окружения NEEDLE_TELEMETRY=0 и DO_NOT_TRACK=1.

## Цитирование

Needle создана командой Cactus Compute. Если вы используете её в своей работе, пожалуйста, цитируйте:

```bibtex
@misc{needle3_2026,
  title        = {Needle: Automation Foundation Model for Tiny Devices},
  author       = {Ndubuaku, Henry and Mosoyan, Karen and Mroz, Jakub and Cylich, Noah and
                  Kumar, Satyajit and Sandhu, Parkirat and Shemet, Roman and Lee, Justin H.},
  year         = {2026},
  organization = {Cactus Compute, Inc.},
  howpublished = {\url{https://github.com/cactus-compute/needle}}
}
```
