# SEF Benchmarks

Questa cartella contiene benchmark piccoli e ripetibili pensati per README e
tesi. L'obiettivo non è dimostrare che SEF è "piu veloce" di ogni alternativa,
ma misurare gli effetti di alcune scelte architetturali: batch, streaming,
materializzazione e piano di esecuzione.

## Note metodologiche

- Ogni scenario viene eseguito in un processo Python separato, cosi la memoria
  RSS di picco è piu confrontabile tra scenari.
- Il tempo misurato esclude l'avvio del processo figlio e misura solo
  `pipeline.run()`.
- I frame sono sintetici per evitare variabilita dovuta a codec, disco,
  webcam o modelli esterni.
- Nel benchmark sulle policy di latenza il processor è volutamente piu lento
  della sorgente. Questo non rappresenta un algoritmo specifico, ma una
  situazione controllata di backpressure.
- Nel benchmark sull'overhead, il baseline diretto non sostituisce gli artifact
  di dominio SEF: evita solo il runtime di pipeline, cosi il confronto rimane
  focalizzato sull'astrazione architetturale.
