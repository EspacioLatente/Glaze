# What Is Glaze?
Glaze is a tool to help artists to prevent their artistic styles from being learned and mimicked by new AI-art models such as MidJourney, Stable Diffusion and their variants. It is a collaboration between the University of Chicago SAND Lab and members of the professional artist community, most notably Karla Ortiz. Glaze has been evaluated via a user study involving over 1,100 professional artists. At a high level, here's how Glaze works:

- Suppose we want to protect artist Karla Ortiz's artwork in her online portfolio from being taken by AI companies and used to train models that can imitate Karla's style.
- Our tool adds very small changes to Karla's original artwork before it is posted online. These changes are barely visible to the human eye, meaning that the artwork still appears nearly identical to the original, while still preventing AI models from copying Karla's style. We refer to these added changes as a "style cloak" and changed artwork as "cloaked artwork." 

![fig1](https://web.archive.org/web/20230322153716im_/https://glaze.cs.uchicago.edu/images/fig1.png)

For example, Stable Diffusion today can learn to create images in Karla's style after it sees just a few pieces of Karla's original artwork (taken from Karla's online portfolio). However, if Karla uses our tool to cloak her artwork, by adding tiny changes before posting them on her online portfolio, then Stable Diffusion will not learn Karla's artistic style. Instead, the model will interpret her art as a different style (e.g., that of Vincent van Gogh). Someone prompting Stable Diffusion to generate "artwork in Karla Ortiz's style" would instead get images in the style of Van Gogh (or some hybrid). This protects Karla's style from being reproduced without her consent. You can read our research paper (currently under peer review). 

![fig2](https://web.archive.org/web/20230322153716im_/https://glaze.cs.uchicago.edu/images/fig2.png)

With Glaze (above), a model trains on cloaked versions of Karla's art, and learns a different style from her original visual style. When it is asked to mimic Karla, it produces art that is distinctively different from Karla's style. 


# Risks and Limitations

- Changes made by Glaze are more visible on art with flat colors and smooth backgrounds, e.g. animation styles. While this is not unexpected, we are searching for methods to reduce the visual impact for these styles.
- Unfortunately, Glaze is not a permanent solution against AI mimicry. AI evolves quickly, and systems like Glaze face an inherent challenge of being future-proof (Radiya et al). Techniques we use to cloak artworks today might be overcome by a future countermeasure, possibly rendering previously protected art vulnerable. It is important to note that Glaze is not panacea, but a necessary first step towards artist-centric protection tools to resist AI mimicry. We hope that Glaze and followup projects will provide some protection to artists while longer term (legal, regulatory) efforts take hold.
- Despite these risks, we designed Glaze to be as robust as possible, and have tested it extensively against known systems and countermeasures. To the best of our knowledge, this is the only available tool for artists to proactively protect their style while posting their art online today. Note that this is ongoing research, and we will continuously update our tools to improve its robustness. 


# About UChicago SAND Lab team (originally developed by)

We are an academic research group of PhD students and CS professors interested in protecting Internet users from invasive uses of machine learning.

We are not motivated by profit or any political agenda, and our only goal is to explore how ethical security techniques can be utilized to develop practical solutions and (hopefully) help real users.

The team:

- Shawn Shan
- Jenna Cryan
- Emily Wenger
- Prof. Heather Zheng
- Prof. Ben Zhao (Faculty lead)
- Prof. Rana Hanocka (Collaborator)
- Karla Ortiz (Artist & Collaborator)
- Lyndsey Gallant (Artist & Collaborator)
- Nathan Fowkes (Artist & Collaborator)

You can reach the UChicago SAND Lab team by email: [here](glaze-uchicago@googlegroups.com)



# Some media coverage of our project:

- [New York Times](https://web.archive.org/web/20230322043359/https://www.nytimes.com/2023/02/13/technology/ai-art-generator-lensa-stable-diffusion.html), by Kashmir Hill
- [UChicago CS News](https://web.archive.org/web/20230322205213/https://cs.uchicago.edu/news/uchicago-scientists-develop-new-tool-to-protect-artists-from-ai-mimicry/), by Rob Mitchum
- TechCrunch, by Natasha Lomas
- [Lifewire](https://web.archive.org/web/20230321181318/https://www.lifewire.com/how-artists-are-fighting-back-against-ai-art-that-copies-their-works-7110583), by Sascha Brodsky
- [Tech Times](https://web.archive.org/web/20230321181356/https://www.techtimes.com/articles/287711/20230215/new-anti-ai-art-generators-tool-protect-artists-heres-glaze.htm)
- [The Register (UK)](https://web.archive.org/web/20230321181309/https://www.theregister.com/2023/02/16/computer_scientists_develop_new_technique/), Katyanna Quach
- [Artnet](https://web.archive.org/web/20230321181336/https://news.artnet.com/art-world/ai-art-theft-tool-2256048), Jo Lawson-Tancred
- [DIY Photography](https://web.archive.org/web/20230321181318/https://www.diyphotography.net/scientists-have-created-a-tool-that-stops-ai-stealing-your-photographs/), Alex Baker
- [WTTW News](https://web.archive.org/web/20230321181320/https://news.wttw.com/2023/02/16/uchicago-develops-tool-protect-artists-ai-threat), Paul Caine
- [La Nacion](https://web.archive.org/web/20230321181317/https://www.lanacion.com.ar/agencias/desarrollan-una-aplicacion-que-evita-que-los-modelos-de-ia-aprendan-y-reproduzcan-el-estilo-de-los-nid17022023/), Europa Press
- Xataka, Javier Marquez
- [Tech Register UK](https://web.archive.org/web/20230316012404/https://www.techregister.co.uk/this-could-block-text-to-image-ai-models-from-ripping-off-artists/)
- [NBC News Nigeria](https://web.archive.org/web/20230321181308/https://www.nbcnews.com.ng/new-anti-ai-art-generators-tool-to-protect-artists-heres-how-glaze-software-works/)
- [Heise Online](https://web.archive.org/web/20230322232719/https://www.heise.de/news/Bildgeneratoren-Glaze-soll-Kunst-fuer-KI-unlernbar-machen-7495423.html), Martin Holland
- [Levtech Japan](https://web.archive.org/web/20230321181321/https://levtech.jp/media/article/column/detail_201/), Hiroki Yamashita
- [Redshark](https://web.archive.org/web/20230322153716/https://www.redsharknews.com/time-to-cloak-your-work-to-stop-ai-ripping-it-off), by Phil Rhodes
- [CG Channel](https://web.archive.org/web/20230322153716/https://www.cgchannel.com/2023/03/glaze-protects-artists-images-from-unethical-ai-models/), by Jim Thacker 

# License

- [GNU GENERAL PUBLIC LICENSE Version 3](https://github.com/EspacioLatente/Glaze/blob/main/LICENSE)
