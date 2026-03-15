# Degisiklik Ozeti

Bu iterasyonda `Experiment Studio` sayfasi prompt-aware hale getirildi. Kullanici artik:

- Langfuse'taki mevcut bir prompt adini girerek `task prompt` veya `judge prompt` cözebiliyor,
- prompt bulunamazsa ayni ekranda custom prompt ile fallback edebiliyor,
- `run_experiment(...)` veya `run_batched_evaluation(...)` akisini bu prompt kaynaklariyla calistirabiliyor,
- son calistirdigi run ozetlerini PostgreSQL'de saklayip ayri bir `Run History` sayfasinda inceleyebiliyor.

Sistem ayrica prompt ozetlerini normalize ediyor, tam prompt body veya item-level ham payload'i DB'ye yazmiyor ve sadece konfigurasyon + aggregate metric seviyesi bir gecmis tutuyor.

Bu iterasyona ek olarak iki yeni analytics sayfasi eklendi:

- `Prompt Analytics`: prompt degisikliklerinden sonraki `cost`, `latency`, `input/output/total token` etkisini gosterir.
- `Tool Judge`: kullanicinin girdigi tool isimlerine gore observation metrics ve evaluator score ozetlerini gosterir.
- `Prompt Coach`: sag alt kosede acilan bir ajan widget'i olarak prompt iyilestirme ve judge onerileri verir.

## Eklenen veya Degistirilen Moduller

- `pyproject.toml`
  - `psycopg[binary]` bagimliligi eklendi.
- `src/core/langfuse_client.py`
  - `get_prompt(...)` destegi eklendi.
  - Dataset ve prompt lookup ayni client uzerinden yonetiliyor.
  - Metrics API icin typed query helper'lari eklendi:
    - `get_prompt_analytics(...)`
    - `get_run_prompt_analytics(...)`
    - `get_tool_observation_metrics(...)`
    - `get_tool_evaluator_metrics(...)`
- `src/core/prompt_registry.py`
  - Yeni modul.
  - Langfuse prompt name -> normalized prompt resolution katmanini sagliyor.
  - `text` ve `chat` prompt tiplerini destekliyor.
  - Gerekirse custom prompt fallback yapiyor.
- `src/core/env_loader.py`
  - Yeni modul.
  - Repo kokundeki `.env` dosyasini bir kez okuyup `os.environ` icine yukluyor.
  - Mevcut shell environment degerlerini ezmiyor.
  - Repo kokunde `.env` yoksa `src/.env` fallback'i de destekliyor.
- `src/core/metrics_analytics.py`
  - Yeni modul.
  - Metrics API cevabini normalize ederek `Prompt Analytics` ve `Tool Judge` dataset'lerini uretiyor.
  - Tool name -> tags fallback stratejisini uyguluyor.
  - Prompt analytics icin run history ile hibrit zenginlestirme yapiyor.
- `src/core/prompt_coach_agent.py`
  - Yeni modul.
  - PydanticAI tabanli `Prompt Coach` ajanini calistiriyor.
  - Karar, prompt onerisi, judge guidance ve evaluator tavsiyesi uretiyor.
- `src/core/web_search.py`
  - Yeni modul.
  - Prompt Coach icin DuckDuckGo HTML uzerinden basit web search araci sagliyor.
- `src/core/experiment_runner.py`
  - Prompt resolution surecini execution akisina entegre ediyor.
  - `run_prompt_experiment(...)` ve `run_dataset_reevaluation(...)` sonrasi history kaydi olusturuyor.
  - Failed run durumlarinda da minimum prompt summary ile kayit dusmeye calisiyor.
- `src/core/run_history.py`
  - Yeni modul.
  - PostgreSQL tabanli `experiment_runs` tablosunu bootstrap ediyor.
  - Run ozetini kaydediyor ve son run'lari listeliyor.
  - Test icin `InMemoryRunHistoryStore` sagliyor.
- `src/schemas/experiment_runner.py`
  - Prompt source, prompt type, prompt target, run status ve history modelleri eklendi.
  - `ResolvedPrompt`, `PromptResolutionRequest`, `PromptResolutionResult`,
    `ExperimentRunRecord`, `ExperimentRunHistoryResult` modelleri tanimlandi.
  - Run history kayitlarina `task_prompt_version` ve `judge_prompt_version` alanlari eklendi.
- `src/schemas/metrics_analytics.py`
  - Yeni modul.
  - `PromptAnalyticsDataset`, `ToolJudgeDataset` ve ilgili filter/row/summary modelleri tanimlandi.
- `src/schemas/prompt_coach.py`
  - Yeni modul.
  - Prompt Coach request/response, decision, apply target ve web source modellerini tanimliyor.
- `src/frontend/streamlit_app.py`
  - Bes sayfali navigasyon yapisina gecti:
    - `Judge Explorer`
    - `Experiment Studio`
    - `Prompt Analytics`
    - `Tool Judge`
    - `Run History`
  - Script dogrudan `streamlit run src/frontend/streamlit_app.py` ile calistiginda `src` paketinin bulunabilmesi icin repo kokunu `sys.path` uzerine ekleyen bootstrap eklendi.
  - Global tema koyu ve yuksek kontrastli hale getirildi; ana palet `#191919`, `#750E21`, `#E3651D`, `#BED754` uzerinden yeniden tanimlandi.
  - Streamlit 2025-12-31 sonrasi kaldirilacak `use_container_width` parametresi yerine `width="stretch"` kullanacak sekilde UI cagrilari guncellendi.
- `src/frontend/prompt_coach_widget.py`
  - Yeni modul.
  - Sag alt kosede acilan Prompt Coach widget'ini render ediyor.
  - Onerilen promptu task/judge alanlarina uygulama aksiyonlari sagliyor.
- `src/frontend/pages/experiment_studio.py`
  - `Task Prompt Source` ve `Judge Prompt Source` bloklari eklendi.
  - Langfuse prompt fetch, prompt preview, custom fallback, run execution ve result summary tek sayfada toplandi.
  - `studio-hint` ve sayfa icindeki yardim kutulari yeni palette gore yeniden boyandi.
- `src/frontend/pages/run_history.py`
  - Yeni sayfa.
  - PostgreSQL'deki son run ozetlerini tablo + detay paneli olarak gosteriyor.
  - Status bazli filtre ve renkli status badge gorunumu eklendi.
  - Tablo satirlarina status badge/styling yansitildi.
  - Aggregate metric'lerden uretilen zaman trend grafigi `average_score`, `count`, `failed_items`, `processed_items` serilerini destekleyecek sekilde genisletildi.
- `src/frontend/pages/prompt_analytics.py`
  - Yeni sayfa.
  - Prompt version karsilastirma, zaman trendi ve run history enrichment bloklarini gosteriyor.
- `src/frontend/pages/tool_judge.py`
  - Yeni sayfa.
  - Tool observation metrics ve evaluator score ozetlerini ayni ekranda gosteriyor.
- `src/__init__.py`
  - Paket import edilir edilmez `.env` yukleme bootstrap'i cagiriyor.
- `tests/test_experiment_runner.py`
  - Prompt resolution, custom fallback, run history persistence ve yeni UI helper davranislari test edildi.
- `tests/test_env_loader.py`
  - `.env` parsing ve mevcut environment degiskenlerini ezmeme davranisi test edildi.
  - `src/.env` fallback davranisi test edildi.
- `tests/test_metrics_analytics.py`
  - Metrics API query JSON'u, prompt analytics aggregation, tool tags fallback ve evaluator normalizasyonu test edildi.
- `tests/test_prompt_coach.py`
  - DuckDuckGo URL normalizasyonu, parser davranisi ve prompt apply aksiyonu test edildi.
- `tests/test_run_history_page.py`
  - Run History status filtreleme ve aggregate trend flattening helper'lari test edildi.

## Sistem Nasil Calisir

### 1. Prompt resolution

Kullanici `Experiment Studio` ekraninda her prompt alani icin iki kaynaktan birini secer:

- `Langfuse prompt`
- `Custom prompt`

Langfuse secilirse su alanlar kullanilir:

- `prompt name`
- opsiyonel `prompt label`
- opsiyonel `prompt version`
- `prompt type` (`text` veya `chat`)

Akis:

1. UI `PromptResolutionRequest` uretir.
2. `PromptResolverService` `collector.get_prompt(...)` cagirir.
3. Prompt bulunursa `ResolvedPrompt` modeline normalize edilir.
4. Prompt bulunamazsa ve custom prompt varsa fallback edilir.
5. Fallback yoksa execution hataya duser.

`chat` prompt tipinde mesajlar normalize edilip tek bir derlenmis string gorunumu de uretilir. Boylece hem preview hem execution tarafinda ortak bir format kullanilir.

### 2. Prompt Runner

`Prompt Runner` modunda:

1. Dataset item'lari Langfuse'tan isimle cekilir.
2. Task prompt resolve edilir.
3. Judge prompt resolve edilir.
4. `task(...)` fonksiyonu item input + task model + task prompt ile output uretir.
5. Her metric icin evaluator judge modeli ve judge prompt ile puanlama yapar.
6. Langfuse `run_experiment(...)` cagrisi calisir.
7. Sonuc normalize edilir.
8. Aggregate metric ozetleri ve prompt summary bilgileri PostgreSQL history tablosuna yazilir.

### 3. Re-evaluate Existing

`Re-evaluate Existing` modunda:

1. Dataset item'larindaki `source_trace_id` veya `source_observation_id` alanlari okunur.
2. Secilen `scope` (`traces` / `observations`) icin uygun item'lar secilir.
3. Judge prompt resolve edilir.
4. Langfuse `run_batched_evaluation(...)` cagrisi hedef entity ID'leri ile tetiklenir.
5. Sonuclar normalize edilir.
6. Aggregate metric ozeti ve konfig bilgisi PostgreSQL'e yazilir.

### 4. PostgreSQL run history

`run_history.py` ilk calismada `experiment_runs` tablosunu olusturur.

Kaydedilen alanlar:

- `id`
- `created_at`
- `mode`
- `dataset_name`
- `run_name`
- `description`
- `status`
- `task_prompt_source`
- `task_prompt_name`
- `task_prompt_version`
- `task_prompt_type`
- `task_prompt_fingerprint`
- `judge_prompt_source`
- `judge_prompt_name`
- `judge_prompt_version`
- `judge_prompt_type`
- `judge_prompt_fingerprint`
- `task_model`
- `judge_model`
- `metric_names`
- `aggregate_metrics`
- `processed_items`
- `failed_items`
- `dataset_run_id`
- `dataset_run_url`
- `warnings`
- `errors`

Ozellikle kaydedilmeyenler:

- tam custom prompt body
- tam item-level payload
- tam output govdesi

Bu sayede run history ekrani operasyonel ozet verebiliyor ama gereksiz veri sisirmiyor.

### 5. Prompt Analytics

`Prompt Analytics` sayfasi Metrics API `observations` view'u uzerinden prompt bazli aggregate veri ceker.

Ana gorunumler:

1. `promptName + promptVersion` karsilastirma tablosu
2. zaman serisi trend tablosu
3. PostgreSQL run history ile hibrit enrichment tablosu

Gosterilen ana metrikler:

- observation count
- avg latency
- total cost
- input tokens
- output tokens
- total tokens

### 6. Tool Judge

`Tool Judge` sayfasi iki asamali veri toplar:

1. `observations` view ile tool bazli metrics
2. `scores-numeric` ve `scores-categorical` ile evaluator skor ozetleri

Tool esleme stratejisi:

1. once `observation.name`
2. sonuc yoksa `tags`

Varsayilan evaluator presetleri:

- `rag`
- `embedding`
- `retrieval`
- `rerank`
- `tool-call`

### 7. Prompt Coach

`Prompt Coach` widget'i tum sayfalarda sag alt kosede acilir.

Yetenekler:

- kullanici degisiklik istegini yazar
- mevcut task/judge prompt context'i ile birlikte ajan calisir
- `approve / revise / reject` karari verir
- gerekiyorsa yeni bir prompt onerir
- judge guidance ve evaluator tavsiyesi uretir
- dis veri kaynagi gerektiginde web search aracini kullanabilir

Onerilen prompt dogrudan task veya judge custom prompt alanina uygulanabilir.

## Kisa Kod Ornekleri

### Langfuse prompt cözme

```python
from src.core.prompt_registry import resolve_prompt
from src.schemas.experiment_runner import (
    PromptResolutionRequest,
    PromptSource,
    PromptTarget,
    PromptType,
)

result = resolve_prompt(
    PromptResolutionRequest(
        source=PromptSource.LANGFUSE_PROMPT,
        target=PromptTarget.JUDGE,
        prompt_name="judge-support-v2",
        prompt_type=PromptType.TEXT,
        custom_prompt="Fallback judge prompt",
    )
)

print(result.resolved_prompt.source)
print(result.resolved_prompt.compiled_text)
```

### Prompt Runner cagirimi

```python
from src.core.experiment_runner import run_prompt_experiment
from src.schemas.experiment_runner import (
    EvaluatorMetricSpec,
    ExperimentExecutionRequest,
    ExperimentMode,
    PromptSource,
    PromptType,
)

result = run_prompt_experiment(
    ExperimentExecutionRequest(
        dataset_name="support-dataset",
        mode=ExperimentMode.PROMPT_RUNNER,
        task_model="<task-model>",
        judge_model="<judge-model>",
        metrics=[
            EvaluatorMetricSpec(name="helpfulness"),
            EvaluatorMetricSpec(name="correctness"),
        ],
        task_prompt_source=PromptSource.LANGFUSE_PROMPT,
        task_prompt_name="support-task-prompt",
        task_prompt_type=PromptType.TEXT,
        judge_prompt_source=PromptSource.CUSTOM_PROMPT,
        judge_prompt="Cevaplari yardimseverlik ve dogruluk acisindan 0.0-1.0 araliginda puanla.",
    )
)

print(result.status)
print(result.history_record_id)
```

### Run history listeleme

```python
from src.core.experiment_runner import list_recent_experiment_runs

history = list_recent_experiment_runs(limit=10, dataset_name="support-dataset")
for record in history.records:
    print(record.run_name, record.status, record.metric_names)
```

### Streamlit arayuzu

```bash
streamlit run src/frontend/streamlit_app.py
```

## Konfigurasyon

Gerekli environment degiskenleri:

- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST`
- `LANGFUSE_PROJECT_ID` (opsiyonel)
- `EXPERIMENT_TASK_MODEL`
- `EXPERIMENT_JUDGE_MODEL`
- `PROMPT_COACH_MODEL` (opsiyonel; yoksa `EXPERIMENT_JUDGE_MODEL` fallback)
- `DATABASE_URL` veya `POSTGRES_DSN`

Notlar:

- `.env` dosyasi repo kokune (`/Users/oguz/Desktop/evalpipeline/.env`) konulabilir; uygulama bu dosyayi otomatik yukler.
- Repo kokunde `.env` yoksa `/Users/oguz/Desktop/evalpipeline/src/.env` de okunur.
- PostgreSQL tanimli degilse run history persistence skip edilir ve warning doner.
- Prompt resolution tarafinda `label` ve `version` ayni anda verilemez.
- Prompt name sadece Langfuse prompt kaynagi secildiginde zorunludur.

## Mevcut Durum

- Judge explorer sayfasi korunuyor.
- Experiment Studio prompt-aware hale getirildi.
- Task ve judge prompt icin Langfuse prompt name + custom fallback birlikte destekleniyor.
- `text` ve `chat` prompt tipleri normalize ediliyor.
- Prompt Runner ve Re-evaluate Existing akislari history kaydi uretiyor.
- PostgreSQL tabanli run history servisi eklendi.
- `Run History` sayfasi son run ozetlerini gosteriyor.
- `Prompt Analytics` sayfasi eklendi.
- `Tool Judge` sayfasi eklendi.
- Metrics API tabanli analytics servis katmani eklendi.
- Prompt Coach agent ve sag alt widget eklendi.
- Run History sayfasina status bazli filtre ve aggregate trend grafigi eklendi.
- `python3 -m unittest discover -s tests -v` basarili.
- `python3 -m compileall src tests` basarili.
- `python3 -m compileall src/frontend` ile Streamlit sayfa derlemesi tekrar dogrulandi.
- `./.venv/bin/python` ile `streamlit`, `psycopg` ve yeni sayfalar import edildi.
- Gecici `.env` dosyasi ile `LANGFUSE_PUBLIC_KEY` ve `LANGFUSE_SECRET_KEY` yuklemesi dogrulandi.
- `streamlit_app.py` dosyasi repo disi bir `cwd` altindan yuklenerek `ModuleNotFoundError: No module named 'src'` hatasinin giderildigi dogrulandi.
- Frontend CSS derlemesi `python3 -m compileall src/frontend` ile dogrulandi.

## Sonraki Onerilen Adimlar

1. Gercek Langfuse projesinde prompt name lookup davranisini `text` ve `chat` promptler icin canli test edin.
2. `run_batched_evaluation(...)` filter semantiğini canli observation/trace verisinde dogrulayin.
3. PostgreSQL icin migration altyapisi veya schema versioning ekleyin.
4. Run History sayfasina status bazli renkli filtre ve aggregate trend grafigi ekleyin.

## 2026-03-09 Env Varsayilanlari
- `src/.env` icindeki varsayilan model ayarlari OpenAI providerina cekildi.
- Guncellenen alanlar:
  - `EXPERIMENT_TASK_MODEL=openai:gpt-4.1-mini`
  - `EXPERIMENT_JUDGE_MODEL=openai:gpt-4.1`
  - `PROMPT_COACH_MODEL=openai:gpt-4.1`
- Not: Gercek `OPENAI_API_KEY` degeri cevaplarda tekrar edilmez; uygulamanin kullanabilmesi icin `.env` dosyasinda kayitli olmasi gerekir.

## 2026-03-09 Prompt Coach RunContext Duzeltmesi
- `Prompt Coach` calistiginda gorulen `name 'RunContext' is not defined` hatasi giderildi.
- `src/core/prompt_coach_agent.py` icinde `Agent` ve `RunContext` importlari fonksiyon ici kapsamdan modol seviyesine tasindi.
- Koken neden: `pydantic_ai` tool annotation cozumlemesi sirasinda `RunContext` global kapsamda bulunamiyordu.
- Regresyon testi eklendi: `tests/test_prompt_coach.py`
- Dogrulama:
  - `python3 -m unittest tests.test_prompt_coach -v`
  - `python3 -m compileall src/core/prompt_coach_agent.py tests/test_prompt_coach.py`

## 2026-03-09 Prompt Coach Apply State Duzeltmesi
- `Prompt Coach` icinde `Apply to task/judge` basildiginda gorulen `StreamlitAPIException: session_state ... cannot be modified after the widget ... is instantiated` hatasi giderildi.
- `src/frontend/prompt_coach_widget.py` icinde dogrudan widget keylerini degistirmek yerine `studio_pending_prompt_apply` kuyrugu kullanilmaya baslandi.
- `src/frontend/pages/experiment_studio.py` sayfa renderinin basinda bu pending degisiklikleri consume ederek ilgili prompt alanlarini guvenli sekilde guncelliyor.
- Ek olarak kullaniciya bir kez gosterilen basari mesaji eklendi: `studio_prompt_apply_message`.
- Testler guncellendi: `tests/test_prompt_coach.py`
- Dogrulama:
  - `python3 -m unittest tests.test_prompt_coach -v`
  - `python3 -m unittest discover -s tests -v`

## 2026-03-09 Langfuse Batch Filter Duzeltmesi
- `run_batched_evaluation(...)` icin gonderilen `filter` payload'i yanlis bicimdeydi; API `filter` alaninda JSON array beklerken object aliyordu.
- `src/core/experiment_runner.py` icinde yeni `_build_batch_filter(...)` yardimcisi eklendi.
- Yeni davranis:
  - tek ID varsa: `[{"type":"string","column":"id","operator":"=","value":"..."}]`
  - coklu ID varsa: `[{"type":"stringOptions","column":"id","operator":"any of","value":[...]}]`
- Guncellenen testler: `tests/test_experiment_runner.py`
- Dogrulama:
  - `python3 -m unittest tests.test_experiment_runner -v`
  - `python3 -m compileall src/core/experiment_runner.py tests/test_experiment_runner.py`

## 2026-03-09 Reevaluation Mapper Signature Duzeltmesi
- `run_dataset_reevaluation` icindeki mapper imzasi Langfuse `MapperFunction` contract'ina uyumlu hale getirildi.
- Eski durum: `def mapper(entity: Any)`
- Yeni durum: `def mapper(*, item: Any, **kwargs: Any) -> EvaluatorInputs`
- Boylece `run_batched_evaluation(...)` tarafinin `mapper(item=...)` cagrisinda gorulen `unexpected keyword argument 'item'` hatasi giderildi.
- Mapper icindeki veri esleme mantigi korunuyor:
  - `id`, `input`, `output`, `metadata` item uzerinden okunuyor
  - dataset `expected_output` baglaniyor
  - `dataset_item_id` ve `dataset_item_metadata` metadata enrichment korunuyor
- Test double da gercek SDK davranisina yaklastirildi; fake collector mapper'i `item=` ve ek kwargs ile cagiriyor.
- Guncellenen testler: `tests/test_experiment_runner.py`
- Dogrulama:
  - `python3 -m unittest tests.test_experiment_runner -v`
  - `python3 -m unittest discover -s tests -v`

## 2026-03-09 Async Evaluator ve Task Duzeltmesi
- `run_batched_evaluation(...)` sirasinda gorulen `This event loop is already running` hatasi giderildi.
- Koken neden: PydanticAI `Agent.run_sync()` cagrilari Langfuse'in async evaluator/task yurutumu icinden calistiriliyordu.
- `src/core/experiment_runner.py` icinde async uyumlu yollar eklendi:
  - `LLMGateway.agenerate_task_output(...)`
  - `LLMGateway.aevaluate_metric(...)`
  - `PydanticAIGateway` icinde `Agent.run(...)` kullanan async implementasyonlar
- `Prompt Runner` task fonksiyonu ve custom evaluator fonksiyonlari async hale getirildi.
- Boylece hem experiment task generation hem de judge evaluator akislari nested event loop acmadan calisiyor.
- Test double'lar coroutine donuslerini dogru handle edecek sekilde guncellendi: `tests/test_experiment_runner.py`
- Dogrulama:
  - `python3 -m unittest tests.test_experiment_runner -v`
  - `python3 -m unittest discover -s tests -v`

## 2026-03-09 Experiment Studio Detail Render Duzeltmeleri
- `Result detail` panelinde duz string veriler `st.json(...)` ile render edildigi icin JSON parse hatalari goruluyordu.
- `src/frontend/pages/experiment_studio.py` icine type-aware `_render_value_block(...)` yardimcisi eklendi.
- Yeni davranis:
  - `dict/list` ise `st.json(...)`
  - JSON parse edilebilen string ise `st.json(...)`
  - duz string ise disabled `st.text_area(...)`
  - `None` ise `Veri yok.` gosterimi
- `Evaluations` bolumu bos oldugunda artik bos tablo yerine acik bilgi mesaji gosteriliyor.
- `src/core/experiment_runner.py` tarafinda reevaluation mapper entity snapshot bilgilerini tutuyor; boylece `output` ve `trace_id` sonuc satirlarina tasiniyor.
- Guncellenen testler:
  - `tests/test_experiment_runner.py`
  - `tests/test_experiment_studio_page.py`
- Dogrulama:
  - `python3 -m unittest tests.test_experiment_runner tests.test_experiment_studio_page -v`
  - `python3 -m unittest discover -s tests -v`

## 2026-03-09 Langfuse Prompt Publish + Judge Reuse Flow
- `Experiment Studio` icine prompt publish ve published prompt reuse akisi eklendi.
- Langfuse SDK yerel surumunde dogrulanan yuzeyler kullanildi:
  - `create_prompt(...)`
  - `update_prompt(...)`
  - `get_prompt(...)`
- Yeni backend yetenekleri:
  - `publish_prompt(...)`
  - `run_llm_judge_on_existing_results(...)`
  - published task/judge prompt metadata'sini request/result/history katmanlarina tasima
- `PromptResolverService` artik sadece prompt okumuyor, ayni prompt adi altinda yeni versiyon publish edebiliyor.
- `ExperimentExecutionRequest` ve `ExperimentExecutionResult` published prompt alanlari ile genisletildi.
- Judge evaluator metadata'sina su alanlar eklendi:
  - `judge_prompt_name`
  - `judge_prompt_label`
  - `judge_prompt_version`
  - `judge_prompt_fingerprint`
  - `judge_prompt_source`
- PostgreSQL `experiment_runs` tablosu additive sekilde genisletildi:
  - `task_prompt_label`
  - `judge_prompt_label`
  - `published_from_custom`
  - `published_at`
- `Run History` sayfasi artik prompt label ve publish bilgisini de gosteriyor.
- `Experiment Studio` akisi:
  1. promptu custom veya Langfuse kaynagindan resolve et
  2. isterse `Publish task prompt` / `Publish judge prompt`
  3. `Use published version in next run` ile published versiyonu sec
  4. `Run experiment` veya reevaluation calistir
- Kisa kod ornegi:
```python
publish_result = publish_prompt(
    PublishedPromptRequest(
        target=PromptPublishTarget.JUDGE,
        prompt_name="judge-support",
        prompt_type=PromptType.TEXT,
        prompt_text="Judge support answers strictly.",
        label="production",
    )
)
```
- Testler eklendi/guncellendi:
  - `tests/test_experiment_runner.py`
  - `tests/test_experiment_studio_page.py`
- Dogrulama:
  - `python3 -m unittest tests.test_experiment_runner tests.test_experiment_studio_page tests.test_run_history_page -v`
  - `python3 -m unittest discover -s tests -v`
- Mevcut durum:
  - publish + reuse backend olarak aktif
  - UI akisi aktif
  - run history publish metadata tutuyor
- Sonraki mantikli adimlar:
  - published prompt list/search secimi icin ayri Langfuse prompt browser eklemek
  - chat prompt editoryal deneyimini text area yerine structured message editor ile iyilestirmek

## 2026-03-09 Published Prompt Browser + Structured Chat Editor
- `Experiment Studio` icine ayri Langfuse published prompt browser eklendi.
- Browser akisi:
  - name/label ile prompt arama
  - Langfuse `prompts.list(...)` uzerinden sonuclarin cekilmesi
  - type (`text` / `chat`) bazli filtreleme
  - secili promptun form alanlarina guvenli sekilde uygulanmasi
- Browser secimi icin `studio_pending_prompt_browser_selection` kuyrugu eklendi; boylece Streamlit widget state hatasi olmadan prompt alanlari dolduruluyor.
- `src/core/langfuse_client.py` ve `src/core/prompt_registry.py` icine prompt listeleme helper'lari eklendi.
- `Chat Prompt Editor` eklendi:
  - role/content tabanli structured editor
  - roller: `system`, `user`, `assistant`, `placeholder`
  - dynamic row ekleme/silme
  - compiled preview gostergesi
- Chat helper davranislari:
  - bos satirlar filtreleniyor
  - message listesi string preview'a donusturuluyor
- Guncellenen moduller:
  - `src/frontend/pages/experiment_studio.py`
  - `src/core/langfuse_client.py`
  - `src/core/prompt_registry.py`
  - `src/core/experiment_runner.py`
- Testler eklendi/guncellendi:
  - `tests/test_experiment_studio_page.py`
- Dogrulama:
  - `python3 -m unittest tests.test_experiment_runner tests.test_experiment_studio_page tests.test_run_history_page -v`
  - `python3 -m unittest discover -s tests -v`
- Mevcut durum:
  - published promptlar artik listeden secilebiliyor
  - chat promptlar duz text area yerine structured editor ile duzenlenebiliyor
- Sonraki mantikli adim:
  - Langfuse prompt detaylarini (config/tags/last_updated) ayri drawer ile gostermek

## 2026-03-09 Prompt Coach + Langfuse MCP Entegrasyonu
- Prompt Coach artik Langfuse MCP server'a baglanabiliyor.
- Yeni modül eklendi: `src/core/langfuse_mcp.py`
  - `LANGFUSE_HOST` uzerinden MCP endpoint kuruyor: `/api/public/mcp`
  - `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` ile Basic Auth header uretiyor
  - `pydantic_ai.mcp.MCPServerStreamableHTTP` toolset'i donuyor
- `src/core/prompt_coach_agent.py` guncellendi:
  - Agent artik MCP toolset ile kuruluyor
  - system prompt, MCP mevcutsa once mevcut Langfuse promptlarini incelemesini istiyor
  - task/judge prompt name ve label bilgileri de coach request promptuna ekleniyor
- `src/schemas/prompt_coach.py` guncellendi:
  - `current_task_prompt_name`
  - `current_judge_prompt_name`
  - `current_task_prompt_label`
  - `current_judge_prompt_label`
- `src/frontend/prompt_coach_widget.py` guncellendi:
  - Studio state'teki mevcut Langfuse prompt name/label bilgileri Prompt Coach request'ine aktariliyor
- Prompt Coach artik apply oncesi mevcut Langfuse promptlari MCP uzerinden gorup kiyaslayabilecek altyapiya sahip.
- Testler guncellendi:
  - `tests/test_prompt_coach.py`
  - MCP endpoint/auth helper testleri eklendi
  - Agent'in MCP server'i toolset olarak aldigi dogrulandi
- Dogrulama:
  - `python3 -m unittest tests.test_prompt_coach -v`
  - `python3 -m unittest discover -s tests -v`
- Not:
  - Bu entegrasyonun canli calismasi icin `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` env'lerinin dolu olmasi gerekir.

## 2026-03-09 Prompt Coach MCP Prompt Version UI

Bu iterasyonda `Prompt Coach` widget'i icine MCP tarafinda gorunen mevcut prompt versiyonlarini gosteren ayri bir kucuk UI paneli eklendi.

### Ne degisti

- `src/frontend/prompt_coach_widget.py`
  - `MCP-visible prompt versions` adli ayri bir `expander` eklendi.
  - Panel, aktif `task` ve `judge` prompt name referanslarini ayri olarak gosteriyor.
  - `Load visible prompt versions` butonu ile Langfuse prompt listesi istege bagli yukleniyor.
  - Yuklenen satirlar icin hedef bazli ozet metrikleri (`task` ve `judge` row count) gosteriliyor.
  - Prompt isimleri degistiginde onceki liste ekranda kalmasin diye stale state temizleme mantigi eklendi.
- `tests/test_prompt_coach.py`
  - Gorunen prompt satirlarini olusturma, hedef bazli ozetleme ve stale-state temizligi test edildi.

### Nasil calisir

1. Kullanici `Prompt Coach` panelini acar.
2. Widget mevcut `studio_task_prompt_name` ve `studio_judge_prompt_name` state degerlerini okur.
3. `MCP-visible prompt versions` panelinde aktif referanslar ozetlenir.
4. Kullanici butona basarsa `list_prompts(...)` ile ilgili prompt kayitlari cekilir.
5. Sonuclar tablo halinde gosterilir:
   - `target`
   - `name`
   - `type`
   - `versions`
   - `labels`
   - `tags`
   - `last_updated_at`
6. Prompt referansi degisirse eski liste temizlenir ve yanlis prompt versiyonu gosterilmez.

### Kisa kod ornegi

```python
current_refs = _collect_current_prompt_refs(st.session_state)
_clear_stale_visible_prompt_versions(st.session_state, current_refs)
prompt_rows = list_prompts(name=ref["name"], limit=10)
rows.extend(_build_visible_prompt_rows(ref["target"], prompt_rows))
```

### Durum

- Prompt Coach artik mevcut task/judge prompt adlari icin MCP tarafinda gorunen prompt versiyonlarini ayri bir mini UI ile gosterebiliyor.
- Listeleme istege bagli oldugu icin widget her render'da gereksiz API cagrisi yapmiyor.
- Prompt isimleri degistiginde eski liste state'i temizlendigi icin stale veri riski azaltildi.

### Sonraki adim onerileri

- `versions` alanini secilebilir hale getirip belirli bir versiyonu dogrudan apply/publish akisina baglamak.
- `last_updated_at` ve `labels` icin daha zengin badge gorunumu eklemek.

## 2026-03-09 Dataset Builder Page

Bu iterasyonda Langfuse trace ve score verilerinden yeni dataset olusturmak icin ayri bir `Dataset Builder` sayfasi eklendi.

### Ne degisti

- `src/schemas/dataset_builder.py`
  - Dataset builder akisi icin typed modeller eklendi:
    - `DatasetMetricThreshold`
    - `DatasetBuilderFilters`
    - `DatasetCandidateScore`
    - `DatasetCandidateTrace`
    - `DatasetCandidateResult`
    - `DatasetCreationRequest`
    - `DatasetCreationResult`
- `src/core/langfuse_client.py`
  - Yeni helper'lar eklendi:
    - `get_trace(...)`
    - `create_dataset(...)`
    - `create_dataset_item(...)`
- `src/core/dataset_builder.py`
  - Yeni servis.
  - Score threshold'a gore candidate trace preview uretiyor.
  - `All metrics` mantigiyla tum secili metric esiklerini gecen trace'leri seciyor.
  - `Trace IO` varsayilanini kullanarak dataset item payload'ini uretiyor.
  - Trace input/output eksikse observation fallback deniyor.
  - Langfuse `create_dataset(...)` ve `create_dataset_item(...)` ile dataset olusturuyor.
  - Mevcut dataset adi varsa create islemini blokluyor.
- `src/frontend/pages/dataset_builder.py`
  - Yeni Streamlit sayfasi.
  - 4 ana blok sunuyor:
    - `Dataset Target`
    - `Score Filters`
    - `Candidate Preview`
    - `Create Dataset`
  - Trace detay panelinde input/output ve score summary gosteriliyor.
- `src/frontend/streamlit_app.py`
  - Sidebar navigasyona `Dataset Builder` eklendi.
- `tests/test_dataset_builder.py`
  - Service ve page helper testleri eklendi.

### Nasil calisir

1. Kullanici dataset adi ve opsiyonel metadata girer.
2. Preset veya custom metric secer.
3. Her metric icin `min score` ve opsiyonel `judge name` girer.
4. Sistem secilen her metric icin Langfuse score kayitlarini ceker.
5. Score'lar trace bazinda gruplanir.
6. Yalnizca tum metric threshold kosullarini saglayan trace'ler aday olur.
7. Her aday trace icin `trace input` ve `trace output` bulunur.
8. Create asamasinda:
   - dataset adi mevcutsa islem bloklanir
   - degilse dataset acilir
   - trace'ler dataset item olarak eklenir

### Kisa kod ornegi

```python
preview = preview_dataset_candidates(
    DatasetBuilderFilters(
        metric_thresholds=[
            DatasetMetricThreshold(metric_name="helpfulness", min_score=0.8),
            DatasetMetricThreshold(metric_name="correctness", min_score=0.85),
        ],
        limit=100,
    )
)

result = create_langfuse_dataset(
    DatasetCreationRequest(
        dataset_name="high-score-traces-v1",
        candidates=preview.candidates,
    )
)
```

### Durum

- `Dataset Builder` sayfasi navigasyonda aktif.
- Candidate preview ve Langfuse dataset create akisi calisiyor.
- Varsayilanlar:
  - coklu metric mantigi: `All metrics`
  - dataset item payload: `Trace IO`
  - isim cakismasi: `Block create`

### Sonraki adim onerileri

- Preview tablosuna manuel trace secimi eklemek.
- Observation-level dataset mapping secenegi eklemek.
- Olusan dataset'i ayni sayfadan `Experiment Studio` icinde dogrudan acma aksiyonu eklemek.

## 2026-03-09 Dataset Builder Secim + Experiment Studio Handoff

Bu iterasyonda `Dataset Builder` sayfasi iki operasyonel iyilestirme kazandi:

- preview sonucundan manuel trace secimi
- olusan dataset'i tek tikla `Experiment Studio` sayfasina tasima

### Ne degisti

- `src/frontend/pages/dataset_builder.py`
  - Preview tablosunun ustune `Selected traces for dataset` coklu secim alani eklendi.
  - Dataset create akisi artik preview'deki tum candidate'lari degil, secili trace listesini kullanuyor.
  - `Selected for create` metrik kutusu eklendi.
  - Create basarili olduktan sonra `Open in Experiment Studio` butonu eklendi.
  - Bu buton dataset adini ve fetch edilmis dataset nesnesini `Experiment Studio` state'ine tasiyor.
- `src/frontend/streamlit_app.py`
  - `pending_active_page` icin guvenli page-switch mekanizmasi eklendi.
  - Boylece mevcut radio widget state'ine dogrudan yazmadan sayfa gecisi yapilabiliyor.
- `tests/test_dataset_builder.py`
  - Secili trace filtreleme helper'i test edildi.
  - Dataset'in `Experiment Studio` state'ine tasinmasi test edildi.
  - `pending_active_page -> active_page` gecisi test edildi.

### Nasil calisir

1. Kullanici candidate preview alir.
2. `Selected traces for dataset` alanindan olusturulacak trace'leri manuel secer.
3. `Create dataset in Langfuse` sadece bu secili trace'lerle calisir.
4. Create basariliysa `Open in Experiment Studio` butonu gorunur.
5. Buton:
   - `studio_dataset_name` state'ini doldurur
   - mumkunse dataset'i hemen fetch eder
   - `pending_active_page = "Experiment Studio"` yazar
   - rerun sonrasinda sayfa otomatik `Experiment Studio`ya gecer

### Kisa kod ornegi

```python
selected_candidates = _selected_candidates_from_state(preview)
st.session_state["dataset_builder_result"] = create_langfuse_dataset(
    DatasetCreationRequest(
        dataset_name=target["dataset_name"],
        candidates=selected_candidates,
    )
)
```

### Durum

- Dataset Builder artik manuel secim destekliyor.
- Olusan dataset tek tikla `Experiment Studio`ya tasinabiliyor.
- Sayfa gecisi Streamlit widget-state hatasina dusmeden yapiliyor.

### Sonraki adim onerileri

- Preview tablosuna `select all / clear all` hizli aksiyonlari eklemek.
- `Experiment Studio`ya geciste yeni dataset ile otomatik `Fetch dataset` yenilemesini daha gorunur basari paneliyle desteklemek.

## 2026-03-09 Langfuse Score API Limit Fix

Bu iterasyonda `Dataset Builder` preview akisi sirasinda alinan Langfuse `score_v_2.get` limit hatasi duzeltildi.

### Ne degisti

- `src/core/langfuse_client.py`
  - `list_scores(...)` artik Langfuse API'ye `limit > 100` gondermiyor.
  - `limit` degeri `100` ile clamp ediliyor.
  - Daha fazla score gerekiyorsa pagination ile sayfa sayfa cekim yapiliyor.
- `tests/test_langfuse_client.py`
  - `list_scores(...)` icin yeni regression testi eklendi.
  - Test, backend'in `score_v_2.get(...)` cagrilarinda `limit > 100` kullanmadigini ve ikinci sayfaya gecebildigini dogruluyor.

### Nasil calisir

1. UI daha yuksek bir candidate limiti gonderebilir.
2. Backend bunu dogrudan Langfuse score endpoint'ine iletmez.
3. `list_scores(...)` her cagrida en fazla `100` kayit ister.
4. Gerekirse `page=2`, `page=3` ile devam eder.
5. Toplanan sonuclar local filter'dan gecirilip istenen limite gore kirpilir.

### Kisa kod ornegi

```python
page_size = max(1, min(filters.limit, 100))
while len(collected) < max(filters.limit, 1):
    response = self.sdk_client.api.score_v_2.get(page=page, limit=page_size, ...)
```

### Durum

- `Dataset Builder` preview sirasinda `limit <= 100` kisiti nedeniyle patlama riski giderildi.
- UI candidate limiti yuksek kalsa bile backend endpoint-uyumlu sekilde pagination yapiyor.

## 2026-03-09 Dataset Builder Timeout Dayanikliligi

Bu iterasyonda `Dataset Builder` preview akisi, Langfuse score fetch timeout'larina karsi daha dayanikli hale getirildi.

### Ne degisti

- `src/core/langfuse_client.py`
  - `LangfuseConfig` icine yeni runtime ayarlari eklendi:
    - `timeout_seconds`
    - `score_page_size`
    - `api_retries`
  - Bu degerler env ile override edilebilir:
    - `LANGFUSE_TIMEOUT_SECONDS`
    - `LANGFUSE_SCORE_PAGE_SIZE`
    - `LANGFUSE_API_RETRIES`
  - Langfuse client artik `timeout=` ile olusturuluyor.
  - `list_scores(...)` artik daha kucuk page size (`50` varsayilan) kullanıyor.
  - `score_v_2.get(...)` icin read-timeout retry eklendi.
  - Her score isteginde `request_options={"timeout_in_seconds": ...}` gonderiliyor.
- `src/core/dataset_builder.py`
  - Metric score fetch hatalari artik sayfayi crash ettirmiyor.
  - Timeout veya benzeri fetch hatalari warning olarak donuyor.
  - Preview sonucu bos da olsa UI ayakta kaliyor.
- `tests/test_langfuse_client.py`
  - read-timeout retry testi eklendi.
- `tests/test_dataset_builder.py`
  - metric fetch hatasi durumunda preview'nin warning ile donecegi test edildi.

### Nasil calisir

1. Dataset Builder preview bir metric icin score ceker.
2. Score fetch timeout olursa backend otomatik retry dener.
3. Retry'lere ragmen basarisiz olursa exception Streamlit'e cikarilmaz.
4. Bunun yerine ilgili metric icin warning uretilir.
5. Diger metricler calismaya devam eder.

### Kisa kod ornegi

```python
response = self.sdk_client.api.score_v_2.get(
    request_options={"timeout_in_seconds": self.config.timeout_seconds},
    **kwargs,
)
```

### Durum

- `ReadTimeout` nedeniyle Dataset Builder sayfasinin dusme riski azaltildi.
- Varsayilan score page size daha kucuk oldugu icin agir sorgularda timeout olasiligi da dustu.
- Tumuyle basarisiz sorgularda bile UI warning verip calismaya devam ediyor.

## 2026-03-15 LiteLLM Cost Dataset Builder

Bu iterasyonda mevcut LiteLLM Postgres log/store verisinden request-level cost dataset olusturmak icin yeni bir `LiteLLM Cost Builder` akisi eklendi.

### Ne degisti

- `src/schemas/litellm_cost_builder.py`
  - LiteLLM cost builder icin typed modeller eklendi:
    - `LiteLLMFieldMapping`
    - `LiteLLMStoreConfig`
    - `LiteLLMCostFilters`
    - `LiteLLMCostCandidateRow`
    - `LiteLLMCostPreviewSummary`
    - `LiteLLMCostDatasetPreview`
    - `LiteLLMCostDatasetRequest`
    - `LiteLLMCostDatasetResult`
- `src/core/litellm_store.py`
  - Yeni Postgres adapter.
  - LiteLLM log tablosunu env tabanli configurable column mapping ile okuyor.
  - Read-only calisiyor; migration yapmiyor.
  - `require_langfuse_join` filtreleme ve cost/token/latency filtrelerini SQL seviyesinde destekliyor.
- `src/core/litellm_cost_builder.py`
  - Yeni servis.
  - LiteLLM request kayitlarini normalize ediyor.
  - Preview summary hesapliyor.
  - Sonucu Langfuse `create_dataset(...)` ve `create_dataset_item(...)` ile dataset olarak yaziyor.
  - Dataset name collision durumunda create islemini blokluyor.
- `src/frontend/pages/litellm_cost_builder.py`
  - Yeni Streamlit sayfasi.
  - `Store Config`, `Cost Filters`, `Preview`, `Create Langfuse Dataset` bloklarini gosteriyor.
  - Request-level satir secimi ve `Open in Experiment Studio` handoff aksiyonu var.
- `src/frontend/streamlit_app.py`
  - Sidebar navigasyona `LiteLLM Cost Builder` eklendi.
- `tests/test_litellm_cost_builder.py`
  - Service, store config ve page helper testleri eklendi.

### Gerekli env degiskenleri

Yeni LiteLLM source mapping env'leri:

- `LITELLM_DATABASE_URL`
- `LITELLM_LOG_TABLE`
- `LITELLM_ID_COLUMN`
- `LITELLM_CREATED_AT_COLUMN`
- `LITELLM_MODEL_COLUMN`
- `LITELLM_COST_COLUMN`

Opsiyonel alanlar:

- `LITELLM_PROVIDER_COLUMN`
- `LITELLM_INPUT_TOKENS_COLUMN`
- `LITELLM_OUTPUT_TOKENS_COLUMN`
- `LITELLM_TOTAL_TOKENS_COLUMN`
- `LITELLM_LATENCY_MS_COLUMN`
- `LITELLM_STATUS_COLUMN`
- `LITELLM_INPUT_COLUMN`
- `LITELLM_OUTPUT_COLUMN`
- `LITELLM_METADATA_COLUMN`
- `LITELLM_LANGFUSE_TRACE_ID_COLUMN`
- `LITELLM_LANGFUSE_OBSERVATION_ID_COLUMN`
- `LITELLM_DB_TIMEOUT_SECONDS`

### Nasil calisir

1. Kullanici LiteLLM store config durumunu yeni sayfada gorur.
2. Tarih/model/provider/status/cost/token/latency filtrelerini girer.
3. Store adapter ilgili LiteLLM request satirlarini Postgres'ten ceker.
4. Service bu satirlari request-level candidate row'lara normalize eder.
5. Kullanici secili request satirlarini dataset icin filtreler.
6. `Create Langfuse Dataset` ile secili request'ler Langfuse dataset item olarak yazilir.
7. Gerekirse dataset tek tikla `Experiment Studio` sayfasina tasinabilir.

### Kisa kod ornegi

```python
preview = preview_litellm_cost_candidates(
    LiteLLMCostFilters(
        model_names=["gpt-4.1"],
        providers=["openai"],
        min_cost=0.01,
        limit=100,
    )
)

result = create_litellm_cost_dataset(
    LiteLLMCostDatasetRequest(
        dataset_name="litellm-cost-requests-v1",
        rows=preview.rows,
        filters=preview.filters,
    )
)
```

### Durum

- LiteLLM icin yeni bir request-level cost dataset builder eklendi.
- Veri kaynagi mevcut Postgres/store; yeni LiteLLM proxy kurulumu bu iterasyonda yok.
- Langfuse join opsiyonel; request kaydi join olmadan da dataset'e girebilir.
- Full test suite gecti.

### Sonraki adim onerileri

- LiteLLM proxy API endpoint'lerinden cost/log ingestion icin ikinci source adapter eklemek.
- Preview tablosuna model/provider toplulaştırma ve grouped cost breakdown kartlari eklemek.
- LiteLLM dataset create sonuclarini Run History veya ayri bir cost history tablosuna yazmak.

## 2026-03-16 Ornek .env Dosyasi

### Ne degisti

- Repo kokune yeni bir `.env.example` dosyasi eklendi.
- Dosya; Langfuse, Experiment Studio, Prompt Coach, PostgreSQL run history ve LiteLLM Postgres source mapping alanlarini birlikte gosteriyor.
- Boylece gercek LiteLLM veritabani ile calismaya baslarken gerekli environment degiskenleri tek yerde gorulebilir hale geldi.

### Eklenen veya degisen moduller

- Yeni dosya: `.env.example`

### Nasil calisir

1. Kullanici `.env.example` dosyasini baz alarak kendi `.env` dosyasini doldurur.
2. Uygulamanin env loader'i once repo kokundeki `.env` dosyasini, sonra gerekirse `src/.env` dosyasini okur.
3. LiteLLM Cost Builder sayfasi `LITELLM_*` mapping alanlarini kullanarak mevcut Postgres tablo/kolonlarini okur.
4. Langfuse ve diger uygulama modulleri ayni `.env` uzerinden calisir.

### Kisa kod ornegi

```bash
cp .env.example .env
```

```env
LITELLM_DATABASE_URL=postgresql://user:password@localhost:5432/litellm
LITELLM_LOG_TABLE=LiteLLM_SpendLogs
LITELLM_ID_COLUMN=request_id
LITELLM_CREATED_AT_COLUMN=startTime
LITELLM_MODEL_COLUMN=model
LITELLM_COST_COLUMN=spend
```

### Durum

- LiteLLM entegrasyonu icin gerekli env yuzeyi artik ornek dosya ile dokumante edildi.
- Kullanici tarafinda yalnizca gercek baglanti bilgileri ve kendi tablo/kolon adlariyla doldurulmasi gerekiyor.

### Sonraki adim onerileri

- Gercek LiteLLM tablo yapisina gore `.env` icindeki opsiyonel kolonlari netlestirmek.
- Gerekirse ikinci bir `.env.litellm.example` varyanti eklemek.

## 2026-03-16 LiteLLM Code-First Tablo Bootstrap

### Ne degisti

- LiteLLM store katmani code-first tablo mantigina gecirildi.
- Artik LiteLLM request log tablosu icin canonical bir Postgres semasi kod icinde tanimli.
- Store, DSN mevcutsa ve `LITELLM_AUTO_CREATE_TABLE=true` ise ilk erisimde `CREATE TABLE IF NOT EXISTS` ile tabloyu bootstrap ediyor.
- `LiteLLM Cost Builder` sayfasina `Ensure LiteLLM table exists` aksiyonu eklendi.
- `.env.example` code-first tablo varsayimlarini gosterecek sekilde guncellendi.

### Eklenen veya degisen moduller

- `src/schemas/litellm_cost_builder.py`
  - Canonical LiteLLM tablo default alanlari eklendi.
  - `LiteLLMStoreConfig` modeline `auto_create_table`, `schema_mode`, `table_bootstrapped` alanlari eklendi.
- `src/core/litellm_store.py`
  - Env verilmezse canonical `litellm_request_logs` tablosuna defaultlayan mapping eklendi.
  - `ensure_schema()` ile code-first tablo ve index bootstrap mantigi eklendi.
  - DDL icin guvenli identifier quoting eklendi.
- `src/core/litellm_cost_builder.py`
  - `ensure_litellm_store_schema()` helper eklendi.
- `src/frontend/pages/litellm_cost_builder.py`
  - Store config bolumune tablo bootstrap butonu ve durum bilgisi eklendi.
- `.env.example`
  - Canonical tablo alanlari ve `LITELLM_AUTO_CREATE_TABLE` eklendi.
- `tests/test_litellm_cost_builder.py`
  - Code-first config, DDL ve index uretimi testleri eklendi.

### Nasil calisir

Canonical tablo varsayimi:

- tablo adi: `litellm_request_logs`
- primary key: `request_id`
- ana kolonlar:
  - `created_at`
  - `model_name`
  - `provider`
  - `total_cost`
  - `input_tokens`
  - `output_tokens`
  - `total_tokens`
  - `latency_ms`
  - `status`
  - `request_input`
  - `request_output`
  - `metadata`
  - `langfuse_trace_id`
  - `langfuse_observation_id`

Akis:

1. Uygulama `LITELLM_DATABASE_URL` veya `DATABASE_URL` ile Postgres baglantisini alir.
2. `PostgresLiteLLMStore` canonical mapping'i kullanir.
3. `LITELLM_AUTO_CREATE_TABLE=true` ise store ilk sorgudan once tablo ve index'leri kontrol eder.
4. Tablo yoksa koddan uretilen DDL ile olusturulur.
5. `LiteLLM Cost Builder` sayfasi ayni tabloyu okuyarak preview ve dataset create akisina devam eder.
6. Mevcut farkli bir LiteLLM tablon varsa `LITELLM_*_COLUMN` env'leri ile override edebilirsin.

### Kisa kod ornegi

```python
store = PostgresLiteLLMStore()
warnings = store.ensure_schema()
rows, warnings = store.list_requests(LiteLLMCostFilters(limit=100))
```

```env
LITELLM_DATABASE_URL=postgresql://user:password@localhost:5432/litellm
LITELLM_AUTO_CREATE_TABLE=true
```

### Durum

- LiteLLM store artik env mapping verilmeden de canonical code-first tablo ile calisabilir.
- Canonical tabloyu olusturma ve index uretimi testlerle dogrulandi.
- Full test suite gecti.

### Sonraki adim onerileri

- Canonical tabloya insert eden ayri bir LiteLLM ingestion writer eklemek.
- Bootstrap edilen tabloya migration/version alanlari ekleyip sema evrimini takip etmek.

## 2026-03-16 Langfuse API Uyumluluk Duzeltmeleri

### Ne degisti

- `Judge Explorer` icin observations v2 cagrisindan artik `parseIoAsJson` parametresi gonderilmiyor.
- `Prompt Analytics` icin Metrics API query zaman damgalari timezone-aware ISO formatina cevrildi.
- Bu degisikliklerle iki adet 400-level Langfuse request hatasi giderildi.

### Eklenen veya degisen moduller

- `src/core/langfuse_client.py`
  - `list_observations()` icinden deprecated `parse_io_as_json=True` kaldirildi.
  - Metrics query icin `_to_langfuse_iso_datetime()` helper eklendi.
- `tests/test_langfuse_client.py`
  - observations v2 cagrisi artik `parse_io_as_json` gondermiyor regression testi eklendi.
  - metrics timestamp serialization icin regression testi eklendi.

### Nasil calisir

1. `Judge Explorer` observations verisini Langfuse v2 endpoint'inden ham `input/output` string alanlariyla alir.
2. Uygulama gerekiyorsa bu alanlari kendi tarafinda parse eder; endpoint'e parse flag gondermez.
3. `Prompt Analytics` filtreleri `datetime` ise Metrics API query'sine `Z` veya timezone offset iceren gecerli ISO datetime olarak serialize edilir.
4. Naive datetime gelirse uygulama bunu UTC kabul eder.

### Kisa kod ornegi

```python
query["fromTimestamp"] = self._to_langfuse_iso_datetime(from_timestamp)
query["toTimestamp"] = self._to_langfuse_iso_datetime(to_timestamp)
```

```python
response = self.sdk_client.api.observations_v_2.get_many(
    limit=filters.limit,
    cursor=filters.cursor,
    from_start_time=filters.from_date,
    to_start_time=filters.to_date,
)
```

### Durum

- `Judge Explorer` icin observations endpoint uyumluluk hatasi giderildi.
- `Prompt Analytics` icin invalid ISO datetime hatasi giderildi.
- Full test suite gecti.

### Sonraki adim onerileri

- Gerekirse Langfuse API v2 uyumluluklari icin diger endpoint parametrelerini de taramak.

## 2026-03-16 LiteLLM Ingestion Writer

### Ne degisti

- Canonical LiteLLM code-first tabloya veri yazmak icin ayri bir ingestion writer eklendi.
- LiteLLM Cost Builder sayfasina JSON tabanli `Ingestion Writer` bolumu eklendi.
- Store katmani artik canonical LiteLLM request row'larini `upsert` edebiliyor.

### Eklenen veya degisen moduller

- Yeni schema: `src/schemas/litellm_ingestion.py`
  - `LiteLLMIngestionRow`
  - `LiteLLMIngestionRequest`
  - `LiteLLMIngestionResult`
- Yeni servis: `src/core/litellm_ingestion.py`
  - `LiteLLMIngestionWriterService`
  - `ingest_litellm_rows(...)`
- `src/core/litellm_store.py`
  - `upsert_requests(...)` eklendi.
  - `request_id` uzerinden `ON CONFLICT DO UPDATE` ile canonical tabloya upsert yapiyor.
- `src/frontend/pages/litellm_cost_builder.py`
  - Ingestion Writer UI eklendi.
  - JSON object veya JSON array kabul ediyor.
- `tests/test_litellm_cost_builder.py`
  - ingestion writer service ve parsing testleri eklendi.

### Nasil calisir

1. Kullanici veya upstream servis canonical LiteLLM request row'larini JSON olarak verir.
2. UI veya servis bu payload'i `LiteLLMIngestionRow` listesine dogrular.
3. Store gerekirse once code-first tabloyu bootstrap eder.
4. Satirlar `request_id` primary key'i ile upsert edilir.
5. Ayni request tekrar gelirse yeni cost/token/metadata degerleri ile guncellenir.
6. Daha sonra `LiteLLM Cost Builder` ayni tabloyu okuyup preview ve dataset create akisina devam eder.

### Kisa kod ornegi

```python
result = ingest_litellm_rows(
    LiteLLMIngestionRequest(
        rows=[
            {
                "request_id": "req-123",
                "model_name": "gpt-4.1-mini",
                "total_cost": 0.018,
                "input_tokens": 120,
                "output_tokens": 45,
                "request_input": {"messages": [{"role": "user", "content": "Hello"}]},
            }
        ]
    )
)
```

### Durum

- LiteLLM tarafinda artik hem code-first tablo bootstrap'i hem de tabloya veri yazan ingestion writer mevcut.
- Full test suite gecti.

### Sonraki adim onerileri

- LiteLLM proxy veya uygulama callback katmanindan dogrudan bu writer'a veri akitan adapter eklemek.
- Ingestion writer icin batch dosya import ve retry kuyruğu eklemek.
