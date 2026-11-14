# Free-model aggregator

One server, two APIs, ten simulated free-tier model providers — an entry for
[make.fun](https://make.fun) competition `free-model-aggregator-01`
("The best free AI model aggregator", $500).

## What it does

`main.py` (Python 3.13, standard library only) serves the OpenAI Chat
Completions API and the Anthropic Messages API, in JSON and in server-sent
events, over the ten providers the trial brief names. For each of the 20
requests it picks the best free model that meets the request's requirements
(tools, vision, JSON mode, minimum context), speaks that provider's own wire
(OpenAI, Anthropic or Gemini), fails over to the next candidate when a
provider refuses (402/429/5xx), and returns the provider's answer in the wire
the client used — text whole, usage counts and stop reason as the provider
reported them.

`python3 main.py --list` prints the catalog document it registers, byte for
byte, and exits 0.

## Routing

Requests are assigned in weight order to the highest-rated free model that
meets them, under a conservative per-provider call cap (each provider's
minimum per-trial budget minus its maximum scheduled failures, from the kit's
published provider table). At run time a refusal moves the request to the
next candidate; a provider that refuses twice in a row is deprioritised, and
a provider this entry has already called up to its published maximum budget
is skipped outright (it cannot have budget left). Every choice is a function
of the brief, the requests and the providers' answers: no wall clock, no
unseeded randomness, so a replay leaves the same state export.

## Local grading

Practice trials (and fresh draws from the sealed generator) score **99**
(quality 99, served 100, wire 100, failover 100, catalog 100; 20/20 valid)
on all three practice instances and three fresh generator draws. Quality 99
is the practical ceiling: a score of 100 would need every request served by
its best-rated model, which the providers' per-trial budgets do not allow.

## License

MIT. See [LICENSE](LICENSE).
