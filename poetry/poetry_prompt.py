CH_TO_EN = """#请将下列文本翻译为英文：
{}
"""

EXTRACT_ENTITIES = """#你是专业的编剧，请提取文本中的画面，并且修改成SD绘画所需的提示词，中间用;隔开：
-请使用
{}
"""

sd_prompt = """
从现在开始你将扮演一个stable diffusion的提示词工程师，你的任务是帮助我设计stable diffusion的文生图提示词。你需要按照如下流程完成工作。
1、我将给你发送一段图片情景，你需要将这段图片情景更加丰富和具象生成一段图片描述。并且按照“【图片内容】具像化的图片描述”格式输出出来；
2、你需要结合stable diffusion的提示词规则，将你输出的图片描述翻译为英语，并且加入诸如高清图片、高质量图片等描述词来生成标准的提示词，提示词为英语，以“【正向提示】提示词”格式输出出来；
3、你需要根据上面的内容，设计反向提示词，你应该设计一些不应该在图片中出现的元素，例如低质量内容、多余的鼻子、多余的手等描述，这个描述用英文并且生成一个标准的stable diffusion提示词，以“【反向提示】提示词”格式输出出来。
4、你需要提示我在生成图片时需要设置的参数以及给我推荐一个使用的模型以及生成这张图片的最优长宽比例，按照“【参数】Sampling method：参数；Sampling steps：参数；CFG Scale：参数；Seed：参数；最优长宽比：参数”的格式输出给我,其中需要注意的是Sampling method参数请在如下列表中选择“Euler a,Euler,LMS,Heun,DPM2,DPM2a,DPM++ 25 a,DPM++ 2M,DPM++ SDE,DPM fast,DPM adaptive,LMS Karras,DPM2 Karras,DPM2 a Karras,DPM++ 2S a Karras,DPM++ 2M Karras,DPM++ SDE Karras,DDIM,PLIMS,UniPC）”。
例如：我发送：一个二战时期的护士。你回复： 
【图片内容】一个穿着二战期间德国护士服的护士，手里拿着一个酒瓶，带着听诊器坐在附近的桌子上，衣服是白色的，背后有桌子。 
【正向提示】A nurse wearing a German nurse's uniform during World War II, holding a wine bottle and a stethoscope, sat on a nearby table with white clothes and a table behind,full shot body photo of the most beautiful artwork in the world featuring ww2 nurse holding a liquor bottle sitting on a desk nearby, smiling, freckles, white outfit, nostalgia, sexy, stethoscope, heart professional majestic oil painting by Ed Blinkey, Atey Ghailan, Studio Ghibli, by Jeremy Mann, Greg Manchess, Antonio Moro, trending on ArtStation, trending on CGSociety, Intricate, High Detail, Sharp focus, dramatic, photorealistic painting art by midjourney and greg rutkowski；【反向提示】cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers),

输入：{}
"""

red_describe_prompt = """
你是专业的小红书文案专家，请根据下属图片的描述生一段美好的小故事，可以参考的元素幸福，积极，可爱，引人注意，充满希望，情感，专注,正能量、诙谐幽默
-不超过30字
-请使用表情
-简短明快
-可以使用不同的不同的文学流派
-一句话描述出来

描述：{}
"""

red_tag_prompt = """
请翻译以下内容为连贯的中文：
翻译后的内容将用来命名文件

用户内容：{}
"""

red_title_prompt = """
你是一个优秀的小红书内容创作者，现在你想根据一组元数据进行创作，规避所有敏感词汇
先对元数据进行初步预处理，去除重复的词汇，去除所有字符，然后仅选第一组元数据按照以下要求和所示例的内容进行输出
要求：
每行之间不要有任何空行，也不要有回车和任何换行符
第一行是我给你提供的元数据，这是一组照片的元数据描述信息，一组照片有6张照片
第二行为小红书描述你将理解这些词组后，对元数据原文进行详细描述。
第三行为小红书标题你将根据理解总结描述内容输出的标题，要求标题不超过20个字，用小红书风格生成笔记的标题，要求20字以内，涵盖表情符号，注意因为本行内容后期要用于文件夹的命名，要符合文件夹命名的规范。
第四行为小红书内容你将根据理解总结描述内容输出的小红书笔记内容要求内容不超过20个字，用小红书风格生成笔记的标题，要求严格要求20字以内，涵盖表情符号，用粉丝第一人称的拟人化的描述，带俏皮色彩，比如有猫猫就说喵喵喵今天还是很开心，比如元数据中有书籍就以第一人称的感觉说，今天你读书了嘛？，给人以代入感体验
第五行为小红书话题你将描述内容拆解成小红书热度比较高的话题标签 20个以内，与元数据相关的标签，不换行进行输出。
所有你生成的内容要求传递给内容消费者正能量，正念，让粉丝阅读后感觉很好，不可以传递负面情绪，不可以传递负面悲伤情绪，杜绝任何负能量词汇，不要出现忧伤类词汇

这是一个参考示例：
元数据：black cat playingvideo games Nintendo Switch PS controller in hand console gamer kitty playful expression curious eyes white whiskers green eyes console screen purple and blue lights cosy environment game room comfy couch gaming setup cat tree carpet furry paws tail flicking ears perked
小红书描述：黑猫手持PS手柄，使用Nintendo Switch，游戏画面显示紫蓝色灯光下的舒适游戏房间，沙发舒适，有猫树、地毯，猫咪表现得玩得高兴，好奇地看着游戏画面，绿色眼睛，白色胡须，耳朵竖立，毛茸茸的爪子在游戏过程中抖动，尾巴摆动。
小红书标题：黑猫Switch游戏时刻🎮✨
小红书内容：黑猫Switch，宅萌必备萌萌哒！
小红书话题：#黑猫# #视频游戏# #Nintendo Switch# #游戏房间# #舒适环境# #沙发# #猫树# #游戏画面# #好奇眼神# #绿眼睛# #胡须# #耳朵竖立# #毛茸茸爪子# #尾巴摆动#
以下是需要进行处理的元数据

标签：{}
"""

ancient_poetry_prompt = """
你是专业的插画师大师，我们需要希望将古诗转化为插图形式。
首先请描述这首 中文诗中的所有意像物象组成的的画面，
然后将其组成一幅详细描述的画面，最后将其整理为 简洁明快的英文绘画提示词prompt。 
请根据以下内容：
-请根据描述的季节
-绘画技巧：画家需要具备绘画技巧，如素描、绘画技法、色彩运用等。
-文学知识：对古代文学和古诗的了解可以帮助艺术家更好地理解和表达古诗的意义和情感。
-色彩理论：了解色彩的运用和搭配可以帮助设计师选择适合的色彩方案，以传达古诗的氛围和情感
-艺术史知识：了解艺术史可以帮助艺术家借鉴和吸收不同艺术流派和风格的元素，丰富作品的表现力。

古诗：{}

Artistic Instructions:

Prompt (in English):

"""

art_poetry_prompt = """
你是专业的场景构建，我们需要希望提取文本内容中的画面景色，用来更好的展示文本内容。
请抽取这首诗词中描绘的所有画面，详细描述这个场景的画面内容，尽量详细，包含背景，景物，颜色，比例，镜头位置，画风等：
-请提取核心的画面，作者想展示的画面，仔细描述该画面
-涉及人物的 需要详细描述人物动作，神态
-涉及景色的 需要详细描述景色内容
-诗词为中国诗词，请描绘符合中国文化的内容：
-请输出合理且描述非常详细的画面，描述画面中可能的元素：
-当画面表达的为一种情感的时候 请连续合适的场景来展示相应的情感内容：
-请按照 画面一、： 画面二、： 画面三、： ... 这样的格式输出
-画面直接要尽可能的相互关联，展示联系，画面元素要齐全

古诗：{}


"""

art_translate_prompt = """
我们需要使用文本描述作画，下面我会给你一段中文描述，请你根据最后一段内容结合前面的内容，将最后的画面描述翻译成合适的英文提示词，请注意：
#1.请只输出英文提示词
#2.输出内容是关于一副画面的，将其修改为适合文生图的提示词
#3.如果涉及人物请添加 高质量的五官，清晰的五官，高质量的画面，这类类似的描述
#4.请尽可能的简洁清晰的描绘画面

<example>
illustrator, anime , realistic ,sketch , 1girl, ,lip, Sweater,order, Blue gradient background, Neon hair,Textured crop, Canadian, (masterpiece,best quality)

(masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2), (1girl:1.4), extreme detailed,(joshua middleton comic cover art:1.1), (Action painting:1.2),(concretism:1.2),theater dance scene,(hypermaximalistic:1.5),colorful,highest detailed

(Glowing ambiance, enchanting radiance, luminous lighting, ethereal atmosphere, mesmerizing glow, evocative hues, captivating coloration, dramatic lighting, enchanting aura),((nude:1)),(boobs naked:0.55),  ink painting, ((moon:1)),  (masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2), (1girl:1.4), extreme detailed,(joshua middleton comic cover art:1.1), (Action painting:1.2),(concretism:1.2),theater dance scene,(hypermaximalistic:1.5),colorful,highest detailed,

dramatic angle,(fluttered detailed color splashs), (illustration),(((1 girl))),(long hair),(rain:0.9),(hair ornament:1.4),there is an ancient palace beside the girl,chinese clothes,(focus on), color Ink wash painting,(color splashing),colorful splashing,(((colorful))),(sketch:0.8), Masterpiece,best quality, beautifully painted,highly detailed,(denoising:0.6),[splash ink],((ink refraction)), (beautiful detailed sky),moon,highly,detaild,(masterpiece, best quality, extremely detailed CG unity 8k wallpaper,masterpiece, best quality, ultra-detailed),(Lycoris radiata)

masterpiece, best quality, highly detailed, sharp focus, dynamic lighting, vivid colors, texture detail, particle effects, storytelling elements, narrative flair, 16k, HDR, subject-background isolation, 2D, (Authentic skin texture:1.3), traditional chinese ink painting,
1 perfect face girl  BREAK 1 handsome face boy , the boy have a long hair, they all are weared (white:1.2) chinese traditional (transparent:1.2)_clothing, they all closed eyes, the boy was stand up at  same height level side of  the girl, the boy's hand is holding the (girl's big chest:1.3), (the boy is (kissing:1.8) the girl:1.7), Stick out tongue, Drooling, (they are looked at each other:1.4), background is sea,

Cinematic Lighting, masterpiece, best quality, highly detailed, sharp focus, dynamic lighting, vivid colors, texture detail, particle effects, storytelling elements, narrative flair, 16k, HDR, subject-background isolation, 2D, (Authentic skin texture:1.3), traditional chinese ink painting,
1 girl, perfect face, perfect hand, Hands clasped together, white chinese traditional (transparent:1.4) clothing, Looking up at the sky, background is snow, Snowflakes fall,

</example>

内容：{}


"""


art_poetry_prompt_v2="""
我们需要使用文本描述作画。下面我会给你一段中文描述，请你根据最后一段内容结合前面的内容，将最后的画面描述翻译成合适的英文提示词。请注意:

#1. 请只输出英文提示词
#2. 输出内容是关于一副画面的，将其修改为适合文生图AI (如Midjourney, Stable Diffusion等)的提示词格式
#3. 如果涉及人物请添加以下描述:
   - 高质量的五官 (high quality facial features)
   - 清晰的五官 (clear facial features) 
   - 高质量的画面 (high quality image)
   - 极其细致的皮肤纹理 (extremely detailed skin texture)
   - 逼真的眼睛 (realistic eyes)
#4. 请尽可能简洁清晰地描绘画面
#5. 添加一些艺术风格、氛围或技术相关的描述词,如:
   - 电影般的 (cinematic)
   - 史诗般的 (epic)
   - 梦幻的 (dreamy)
   - 超现实主义的 (surrealistic) 
   - 印象派的 (impressionistic)
   - 8K分辨率 (8K resolution)
   - HDR (High Dynamic Range)
#6. 可以适当使用权重,如 (keyword:1.2) 来强调某些元素
#7. 使用 BREAK 来分隔不同的场景或元素描述

<example>
masterpiece, best quality, highly detailed, sharp focus, 8K resolution, HDR, (cinematic lighting:1.3), 1 girl, perfect face, (high quality facial features:1.2), (clear facial features:1.2), (extremely detailed skin texture:1.3), (realistic eyes:1.4), white traditional Chinese dress, (transparent fabric:1.2), looking up at sky, clasped hands BREAK background: snowy landscape, falling snowflakes, misty atmosphere, (dreamy:1.1)

(epic fantasy scene:1.2), (best quality:1.4), (masterpiece:1.3), ultra-detailed, 8K, HDR, dramatic lighting, 1 handsome boy, long hair, perfect face, (high quality facial features:1.2), white traditional Chinese clothing BREAK 1 beautiful girl, (perfect face:1.3), (clear facial features:1.2), white semi-transparent dress, (large breasts:1.1) BREAK (kissing passionately:1.4), (tongues touching:1.2), (intimate embrace:1.3), (looking into each other's eyes:1.2) BREAK background: serene sea, sunset sky, (impressionistic style:1.1)

(surrealistic portrait:1.3), (hyperdetailed:1.4), (best quality:1.2), 8K, sharp focus, 1 girl, ethereal beauty, (high quality facial features:1.3), flowing hair, intricate headdress, (glowing skin:1.2) BREAK background: abstract swirls of color, floating objects, (dream-like atmosphere:1.2), (vibrant colors:1.3)
</example>

内容: {}
"""