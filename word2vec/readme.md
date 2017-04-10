# Word2Vec Statistics

### Accuracy Improvement With Respect to Dataset 

![Accuracy Improvement](../data/Word2Vec_Results_Plot.png)

### Accuracy Improvement With Respect to Window Size & Negative Parameter

![Accuracy Improvement](../data/best_window_negative_plot.png)



### Accuracy Improvement Data

<table class="tg">
  <tr>
    <th class="tg-031e"></th>
    <th class="tg-031e"></th>
    <th class="tg-031e" colspan="3">Best Match</th>
    <th class="tg-031e" colspan="3">Opposite</th>
    <th class="tg-yw4l"></th>
    <th class="tg-yw4l"></th>
  </tr>
  <tr>
    <td class="tg-031e"></td>
    <td class="tg-031e">Model</td>
    <td class="tg-031e">Correct</td>
    <td class="tg-031e">TopK</td>
    <td class="tg-031e">Coverage</td>
    <td class="tg-031e">Correct</td>
    <td class="tg-yw4l">TopK</td>
    <td class="tg-yw4l">Coverage</td>
    <td class="tg-yw4l">Avg. Correct</td>
    <td class="tg-yw4l">Avg. Top K</td>
  </tr>
  <tr>
    <td class="tg-031e" rowspan="7">Combined Fire + Crawled</td>
    <td class="tg-031e">100d_w5_n5</td>
    <td class="tg-031e">3.1</td>
    <td class="tg-031e">10.1</td>
    <td class="tg-031e">79.5</td>
    <td class="tg-031e">8.6</td>
    <td class="tg-yw4l">19.6</td>
    <td class="tg-yw4l">61</td>
    <td class="tg-yw4l">5.85</td>
    <td class="tg-yw4l">14.85</td>
  </tr>
  <tr>
    <td class="tg-031e">100d_w10_n25</td>
    <td class="tg-031e">3.6</td>
    <td class="tg-031e">10.2</td>
    <td class="tg-031e">79.5</td>
    <td class="tg-031e">10.4</td>
    <td class="tg-yw4l">24.3</td>
    <td class="tg-yw4l">61</td>
    <td class="tg-yw4l">7</td>
    <td class="tg-yw4l">17.25</td>
  </tr>
  <tr>
    <td class="tg-031e">100d_w15_n40</td>
    <td class="tg-031e">3.7</td>
    <td class="tg-031e">9.6</td>
    <td class="tg-031e">79.5</td>
    <td class="tg-031e">11</td>
    <td class="tg-yw4l">23.7</td>
    <td class="tg-yw4l">61</td>
    <td class="tg-yw4l">7.35</td>
    <td class="tg-yw4l">16.65</td>
  </tr>
  <tr>
    <td class="tg-031e">100d_w25_n50</td>
    <td class="tg-031e">3.3</td>
    <td class="tg-031e">8.8</td>
    <td class="tg-031e">79.5</td>
    <td class="tg-031e">9.1</td>
    <td class="tg-yw4l">25.2</td>
    <td class="tg-yw4l">61</td>
    <td class="tg-yw4l">6.2</td>
    <td class="tg-yw4l">17</td>
  </tr>
  <tr>
    <td class="tg-031e">300d_w5_n5</td>
    <td class="tg-031e">4.5</td>
    <td class="tg-031e">14.5</td>
    <td class="tg-031e">79.5</td>
    <td class="tg-031e">9.9</td>
    <td class="tg-yw4l">24.4</td>
    <td class="tg-yw4l">61</td>
    <td class="tg-yw4l">8.5</td>
    <td class="tg-yw4l">20.8</td>
  </tr>
  <tr>
    <td class="tg-031e">300d_w10_n25</td>
    <td class="tg-031e">4.9</td>
    <td class="tg-031e">13.3</td>
    <td class="tg-031e">78.9</td>
    <td class="tg-031e">12.1</td>
    <td class="tg-yw4l">28.3</td>
    <td class="tg-yw4l">61</td>
    <td class="tg-yw4l">8..5</td>
    <td class="tg-yw4l">20.8</td>
  </tr>
  <tr>
    <td class="tg-031e">300d_w15_n40</td>
    <td class="tg-031e">5.3</td>
    <td class="tg-031e">13.3</td>
    <td class="tg-031e">78.9</td>
    <td class="tg-031e">12.3</td>
    <td class="tg-yw4l">31.1</td>
    <td class="tg-yw4l">61</td>
    <td class="tg-yw4l">8.8</td>
    <td class="tg-yw4l">22.2</td>
  </tr>
  <tr>
    <td class="tg-031e" rowspan="5">Fire</td>
    <td class="tg-031e">100d_w5_n5</td>
    <td class="tg-031e">2.3</td>
    <td class="tg-031e">8.6</td>
    <td class="tg-031e">72.2</td>
    <td class="tg-031e">3.1</td>
    <td class="tg-yw4l">12.1</td>
    <td class="tg-yw4l">38.8</td>
    <td class="tg-yw4l">2.7</td>
    <td class="tg-yw4l">10.35</td>
  </tr>
  <tr>
    <td class="tg-031e">100d_w5_n25</td>
    <td class="tg-031e">2.8</td>
    <td class="tg-031e">8.7</td>
    <td class="tg-031e">72.2</td>
    <td class="tg-031e">4</td>
    <td class="tg-yw4l">13.9</td>
    <td class="tg-yw4l">38.8</td>
    <td class="tg-yw4l">3.4</td>
    <td class="tg-yw4l">11.3</td>
  </tr>
  <tr>
    <td class="tg-031e">100d_w15_n40</td>
    <td class="tg-031e">3.4</td>
    <td class="tg-031e">8.7</td>
    <td class="tg-031e">72.2</td>
    <td class="tg-031e">4.1</td>
    <td class="tg-yw4l">16.2</td>
    <td class="tg-yw4l">38.8</td>
    <td class="tg-yw4l">3.75</td>
    <td class="tg-yw4l">12.45</td>
  </tr>
  <tr>
    <td class="tg-031e">100d_w25_n50</td>
    <td class="tg-031e">2.9</td>
    <td class="tg-031e">7.7</td>
    <td class="tg-031e">72.2</td>
    <td class="tg-031e">4.7</td>
    <td class="tg-yw4l">17.7</td>
    <td class="tg-yw4l">38.8</td>
    <td class="tg-yw4l">3.8</td>
    <td class="tg-yw4l">12.7</td>
  </tr>
  <tr>
    <td class="tg-yw4l">300d_w5_n5</td>
    <td class="tg-yw4l">3.7</td>
    <td class="tg-yw4l">12.1</td>
    <td class="tg-yw4l">72.2</td>
    <td class="tg-yw4l">4.1</td>
    <td class="tg-yw4l">16.4</td>
    <td class="tg-yw4l">38.8</td>
    <td class="tg-yw4l">3.9</td>
    <td class="tg-yw4l">14.25</td>
  </tr>
</table>





| Dataset        | Model        | Best Match Correct | Best Match TopK | Opposite | Opposite TopK |
|----------------|--------------|--------------------|-----------------|----------|---------------|
| Fire + Crawled | 100d_w5_n5   | 3.1                | 10.1            | 8.6      | 19.6          |
| Fire + Crawled | 100d_w10_n25 | 3.6                | 10.2            | 10.4     | 24.3          |
| Fire + Crawled | 100d_w15_n40 | 3.7                | 9.6             | 11       | 23.7          |
| Fire + Crawled | 100d_w25_n50 | 3.3                | 8.8             | 9.1      | 25.2          |
| Fire + Crawled | 300d_w5_n5   | 4.5                | 14.5            | 9.9      | 24.4          |
| Fire + Crawled | 300d_w10_n25 | 4.9                | 13.3            | 12.1     | 28.3          |
| Fire + Crawled | 300d_w15_n40 | 5.3                | 13.3            | 12.3     | 31.1          |
| Fire           | 100d_w5_n5   | 2.3                | 8.6             | 3.1      | 12.1          |
| Fire           | 100d_w10_n25 | 2.8                | 8.7             | 4        | 13.9          |
| Fire           | 100d_w15_n40 | 3.4                | 8.7             | 4.1      | 16.2          |
| Fire           | 100d_w25_n50 | 2.9                | 7.7             | 4.7      | 17.7          |
| Fire           | 300d_w5_n5   | 3.7                | 12.1            | 4.1      | 16.4          |


