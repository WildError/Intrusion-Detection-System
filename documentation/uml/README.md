# UML-діаграми IDS (розділ 2 магістерської роботи)

У цьому каталозі подано сім UML-діаграм, використаних у розділі 2 кваліфікаційної роботи. Структуру схем узгоджено з реалізацією `app.py` і `ids_project/core.py`. Для редагування використовується формат draw.io.

| Рисунок диплома | UML / тип схеми | Перегляд | Редагування |
|---|---|---|---|
| 2.1 | UML package diagram, логічна архітектура | [PNG](arch.png) | [draw.io](arch.drawio) |
| 2.2 | UML use-case diagram | [PNG](usecase.png) | [draw.io](usecase.drawio) |
| 2.3 | UML class diagram | [PNG](class.png) | [draw.io](class.drawio) |
| 2.4 | UML sequence diagram | [PNG](sequence.png) | [draw.io](sequence.drawio) |
| 2.5 | UML activity diagram | [PNG](activity.png) | [draw.io](activity.drawio) |
| 2.6 | UML component diagram | [PNG](component.png) | [draw.io](component.drawio) |
| 2.7 | UML deployment diagram | [PNG](deploy.png) | [draw.io](deploy.drawio) |

Також доступні SVG із тими самими базовими іменами. Файли `.drawio` відкриваються у https://app.diagrams.net/ і дозволяють змінювати текст та елементи.

**Відповідність коду.** `DatasetLoader`, `SchemaValidator`, `PipelineFactory`, `ThresholdSelector`, `ModelTrainer`, `PredictionService` описують фактичні Python-класи. `get_metrics()` — *функція модуля*, а не клас. Компонентна та пакетна діаграми відображають **логічний поділ** функцій, а не стверджують наявність окремих модулів для кожного блоку. Діаграма розгортання показує локальну роботу на одному ПК.
