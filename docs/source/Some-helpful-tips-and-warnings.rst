Never use the same block in two pipelines.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bifrost blocks are meant to be initialized once, used in a Pipeline, and
then destroyed. If you call a Pipeline twice, or use the same block in
two separate pipelines, there will be unpredictable side effects.
