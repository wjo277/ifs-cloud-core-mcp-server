The search ranking is not that great, it still doesn't give me what I would expect when I search for things. One particular annoyance is that entity and views almost always outperform the other file types, it needs to be more balanced.

Do the following:

1. Create a test script that allows you to run search queries against the search API (use the web_ui for this, as it best matches the real life scenario)
2. Populate that test script with typical phrases an IFS business user would ask for (think about what different departments find important and create the query phrases based on that)
3. Query the intelligent search API for each of these query phrases and request the top 100 results
4. Go through each of the results (and preferably inspect their file contents as well) and ask yourself which 5 results would you have wanted at the top of the search results (the ones being the most relevant to the phrase you queried for)
5. Store these results in a separate test that you can then use to benchmark the next step
6. Iterate on the search method and ranking algorithms and try to generalize so that the benchmark test you created in the previous step improves. Don't be afraid to think radically and new.
7. iterate, iterate, iterate...