"""Emotion guideline text used for prompting (zero-/few-shot)."""
EMOTION_DEFINITIONS_ZERO_SHOOT = {
    "Fear": (
        "Fear expresses stress or anxiety towards an app or a particular event. It implies a sense of agitation, expressing that the user is scared about something."
        "\nFear can refer to users complaining or reporting an incident that affects their trust for an app."
        "\nFear also implies users asking for help, showing frustration or anxiety towards a situation they cannot control."
    ),
    "Joy": (
        "Joy expresses excitement or pleasure towards an app, a feature, a release or a user experience in general. It implies a sense of possibility and positiveness."
        "\nJoy reviews express positive aspects about an app."
        "\nUsers also express Joy when they mention what they like about the app, such as features or characteristics."
        "\nJoy can also be linked to positive user experience."
    ),
    "Anger": (
        "Anger expresses fury and rage experienced as a result of the use of an app. It implies a sense of fierceness and hate, implying that the user has encountered an obstacle causing huge disruption."
        "\nAnger is clearly expressed through hate."
        "\nAnger is often transmitted through hate speech, including insults or extremely negative vocabulary."
        "\nFinally, Anger can also be expressed as a complementary emotion when users express fury through writing style, either with capitalization, exclamation marks or emojis."
        "\nAnger and Disgust can also be confused sometimes for the same reason, as they are contiguous emotions according to Plutchnik's Wheel. In this context, when in doubt, we will apply the following distinction:"
        "\n- If the user expresses hate, fury, rage or any anger-related emotion, even if it is also rejecting the use of the app (implicitly or explicitly), then we will consider the review as Anger."
        "\n- If the user does not express any anger-related emotion, then we will consider the review as Disgust."
        "\nWhile the user is clearly stating that the app does not meet its expectations, the use of hate-related vocabulary implies Anger as the predominant emotion."
        "\nThere is no negative or hateful judgement in addition to the rejection of the app by the user"
    ),
    "Sadness": (
        "Sadness expresses disappointment or a sense of loss towards the expectations of the user with respect to the app. It implies a sense of heaviness and severeness, implying a decrease in the average user experience."
        "\nSadness refers to negative statements about the user experience with respect to user expectations."
        "\nSadness also refers to bug reports or faults."
        "\nSadness can also refer to new releases causing disruption in the traditional activity of the user"
    ),
    "Surprise": (
        "Surprise expresses shocked or unexpected emotions towards an app. It implies a sense of unpredictability related to something new, altering user expectations (either positively, negatively or neutral)."
        "\nSurprise can refer to issues encountered during the use of the app that cannot qualify as bugs but did not suit user expectations."
        "\nSurprise can also refer to events or situations the user encounters that do not suit their judgment."
    ),
    "Disgust": (
        "Disgust expresses a lack of trust and rejection towards an app. It implies a sense of bitterness and refusal, implying a judgment from the user that something in the app is wrong, unsafe or not compliant with rules."
        "\nDisgust is clearly expressed when the user claims that they refuse to use the app."
        "\nDisgust can also be expressed by stating that an app is not suited for purpose, either with sarcasm or literally."
        "\nSadness and Disgust can often be confused, as they are contiguous emotions according to Plutchnik's Wheel. In this context, when in doubt, we will apply the following distinction:"
        "\n- If the focus of the review is on disappointment concerning user expectations in terms of functional correctness, usability, efficiency and other quality aspects, then we will consider the review as Sadness."
        "\n- If the focus of the review is on the explicit rejection and distrust of the user for the app or a set of its features, then we will consider the review as Disgust."
    ),
    "Trust": (
        "Trust expresses acceptance and connection from the user to the app. It implies a sense of safeness and warmness, illustrating explicitly the user's willingness to use the app."
        "\nUsers express Trust when they explicitly mention their personal experience with the app in a personal way, reporting how the app satisfies their goals or needs."
        "\nTrust can also be about the user relying on the app to conduct a specific task according to its expectations."
        "\nTrust is also expressed when the user relies on the app over other alternatives in the market."
        "\nFurthermore, Trust is also related to positive praises of security and privacy concerns:"
        "\nJoy and Trust can often be confused, as they are contiguous emotions according to Plutchnik's Wheel. In this context, when in doubt, we will apply the following distinction:"
        "\n- If the user makes no explicit mention of their own experience, limiting the feedback to positive aspects and likes, with no involvement of their expectations or their use of the application, then we will consider the review as Joy."
        "\n- If the user makes explicit praise in the review about their own user experience with the application from a personal point of view, including the satisfaction of their goals and needs, then we will consider the review as Trust."
    ),
    "Anticipation": (
        "Anticipation expresses curiosity and consideration about something unknown or unclear to the user. It implies a sense of alertness and exploration, related to looking ahead to what could come with uncertainty and/or expectation."
        "\nAnticipation can relate to inquiries and questions about particular upcoming changes, also expressed through feature requests or change proposals."
        "\nAdditionally, Anticipation also relates to considerations and questions raised by users about something confusing, not clear, for which they do not know how to proceed."
        "\nWhile they are opposite emotions according to Plutchnik's Wheel, Surprise and Anticipation can often be confused, especially when the sentence implies uncertainty or concerns about something. In this context, when in doubt, we will apply the following distinction:"
        "\n- If the review entails a reactive emotion from the user's point of view (i.e., raised by a particular event or situation, as a reaction to something that happened), then we will consider Surprise."
        "\n- If the review entails a proactive emotion from the user's point of view (i.e., raised by the curiosity and active consideration of the user), then we will consider Anticipation."
        "\nWhile the user expresses uncertainty about something, this is not emerging from a reactive situation from a particular event, rather than being a consideration (i.e., anticipation) of the user towards the future of the app."
        "\nFinally, Anticipation can also be confused with sadness when users ask for a missing feature. In this context, when in doubt, we will apply the following distinction:"
        "\n- If the review expresses explicit disappointment with the missing feature, then we will consider Sadness."
        "\n- If the review expresses as a wish for the feature to be included, with no disappointment connotations, then we will consider Anticipation"
    ),
}

EMOTION_DEFINITIONS_FEW_SHOOT = {
    "Fear": (
        "Fear expresses stress or anxiety towards an app or a particular event. It implies a sense of agitation, expressing that the user is scared about something."
        "\nFear can refer to users complaining or reporting an incident that affects their trust for an app:"
        "\n[Example 1] People are adding fake members via python by scrapping to make their group look huge."
        "\nFear also implies users asking for help, showing frustration or anxiety towards a situation they cannot control:"
        "\n[Example 2] I having Google Pixel 3 phone, but call recording option is not available on this app please help me."
        "\n[Example 3] Facing fast gifs speed problem and gif folder limit reach. Anyone please help"
    ),
    "Joy": (
        "Joy expresses excitement or pleasure towards an app, a feature, a release or a user experience in general. It implies a sense of possibility and positiveness."
        "\nJoy reviews express positive aspects about an app:"
        "\n[Example 1] Minimalistic, but useful app without ads."
        "\nUsers also express Joy when they mention what they like about the app, such as features or characteristics:"
        "\n[Example 2] It's good and great and it's has a meal planner it's quite helpful."
        "\n[Example 3] Excellent task and list management tool"
        "\nJoy can also be linked to positive user experience:"
        "\n[Example 4] Good exercise and good food suggestions morning exercise give a special energy ðŸ˜Š"
    ),
    "Anger": (
        "Anger expresses fury and rage experienced as a result of the use of an app. It implies a sense of fierceness and hate, implying that the user has encountered an obstacle causing huge disruption."
        "\nAnger is clearly expressed through hate:"
        "\n[Example 1] I hate telegram"
        "\n[Example 2] I hate that it disconnects if the app is not running in the background and sometimes it misses to push notifications."
        "\nAnger is often transmitted through hate speech, including insults or extremely negative vocabulary:"
        "\n[Example 3] And you can see from the support replies, all you get is some worthless canned response from some mindless wage slave, at best."
        "\n[Example 4] Stupid Java supporters."
        "\nFinally, Anger can also be expressed as a complementary emotion when users express fury through writing style, either with capitalization, exclamation marks or emojis."
        "\n[Example 5] The food log needs a BARCODE SCANNER!"
        "\nAnger and Disgust can also be confused sometimes for the same reason, as they are contiguous emotions according to Plutchnik's Wheel. In this context, when in doubt, we will apply the following distinction:"
        "\n- If the user expresses hate, fury, rage or any anger-related emotion, even if it is also rejecting the use of the app (implicitly or explicitly), then we will consider the review as Anger."
        "\n- If the user does not express any anger-related emotion, then we will consider the review as Disgust."
        "\nFor example, in the following Anger example:"
        "\n[Example 6] It wasn't what I needed and I absolutely HATE that they don't even tell you the timers are for premium memberships only."
        "\nWhile the user is clearly stating that the app does not meet its expectations, the use of hate-related vocabulary implies Anger as the predominant emotion."
        "\nOn the other hand, in the following Disgust example:"
        "\n[Example 7] Until you change the privacy settings, I won't update"
        "\nThere is no negative or hateful judgement in addition to the rejection of the app by the user"
    ),
    "Sadness": (
        "Sadness expresses disappointment or a sense of loss towards the expectations of the user with respect to the app. It implies a sense of heaviness and severeness, implying a decrease in the average user experience."
        "\nSadness refers to negative statements about the user experience with respect to user expectations:"
        "\n[Example 1] It makes my workflow almost unusable :("
        "\nSadness also refers to bug reports or faults:"
        "\n[Example 2] Just that RCS/Chat function has regular failures unfortunately."
        "\n[Example 3] Stories are replaying and starting over again and again."
        "\nSadness can also refer to new releases causing disruption in the traditional activity of the user:"
        "\n[Example 4] Latest update blocks ability to choose a bus-only routing in London without moving to the premium version."
        "\n[Example 5] Great app but sadly you can't connect my fitness pal and no food/calorie/macro tracker in app ðŸ˜© only thing it's missing."
    ),
    "Surprise": (
        "Surprise expresses shocked or unexpected emotions towards an app. It implies a sense of unpredictability related to something new, altering user expectations (either positively, negatively or neutral)."
        "\nSurprise can refer to issues encountered during the use of the app that cannot qualify as bugs but did not suit user expectations:"
        "\n[Example 1] wanted to go to an event later in the month showed only 3 attendees but rsvp was closed not showing canceled?"
        "\n[Example 2] Another weird issue is I got a transaction SMS from the bank that had the words 'For lost/stolen cards call xxxx' and Truecaller gave a notification with the title 'Stolen Card' and the last 4 digits of my card which was weird"
        "\nSurprise can also refer to events or situations the user encounters that do not suit their judgment:"
        "\n[Example 3] 3 stars for asking me to rate a planner/task app I installed 1hr ago so I don't know how well it works for a few days minimum before asking for review"
    ),
    "Disgust": (
        "Disgust expresses a lack of trust and rejection towards an app. It implies a sense of bitterness and refusal, implying a judgment from the user that something in the app is wrong, unsafe or not compliant with rules."
        "\nDisgust is clearly expressed when the user claims that they refuse to use the app:"
        "\n[Example 1] I find this app frustrating as it has stopped working and not for the first time , today I was on week 7 workout 1 and it stopped working on my 25 minute run so I had to try and gauge myself when to stop, will be using a running watch from now on"
        "\nDisgust can also be expressed by stating that an app is not suited for purpose, either with sarcasm or literally:"
        "\n[Example 2] Might as well just call it 'Weather Radar' and ditch the widget part of the name because its non-functional."
        "\n[Example 3] It is unnecessary for an athletic networking site to wade into politics."
        "\nSadness and Disgust can often be confused, as they are contiguous emotions according to Plutchnik's Wheel. In this context, when in doubt, we will apply the following distinction:"
        "\n- If the focus of the review is on disappointment concerning user expectations in terms of functional correctness, usability, efficiency and other quality aspects, then we will consider the review as Sadness."
        "\n- If the focus of the review is on the explicit rejection and distrust of the user for the app or a set of its features, then we will consider the review as Disgust."
        "\nFor instance, in the following Sadness examples:"
        "\n[Example 4] Alarm routine won't start gargle podcasts"
        "\n[Example 5] My days aren't all exactly the same so today I had a zoom meeting and couldn't access to it from my phone and had to restart it."
        "\nThe user is clearly reporting a disappointment comparing the performance of the app with respect to their expectations, without expressing any additional or more grave judgment about the acceptance of the app by the user."
        "\nHowever, in the following Disgust examples:"
        "\n[Example 6] Until you change the privacy settings, I won't update"
        "\n[Example 7] The only thing you get out of the app are bots, spamming you and trying to get you to a paid site."
        "\nThe user is either reporting a clear rejection of the app or making a strong statement about the app being useless to satisfy any purpose or need."
    ),
    "Trust": (
        "Trust expresses acceptance and connection from the user to the app. It implies a sense of safeness and warmness, illustrating explicitly the user's willingness to use the app."
        "\nUsers express Trust when they explicitly mention their personal experience with the app in a personal way, reporting how the app satisfies their goals or needs:"
        "\n[Example 1] It has really made a big difference in my health and wellness."
        "\n[Example 2] One of the best note-taking I've tried so far."
        "\n[Example 3] To learn GPS navigation is my second favorite option"
        "\nTrust can also be about the user relying on the app to conduct a specific task according to its expectations:"
        "\n[Example 4] I set up a new, independent account on MY email server specifically for this app to connect to, to backup my phone SMS/call logs, etc."
        "\n[Example 5] My favourite features are recurring tasks - brilliant for implementing new habits and calendar where I can schedule future to do things on specific days."
        "\nTrust is also expressed when the user relies on the app over other alternatives in the market:"
        "\n[Example 6] By far the best app I've used for keeping track of macros."
        "\n[Example 7] Although I use My Diary as a journal, & RPG Notes for TTRPG notes, no app I've used beats this one when a simple & efficient note pad is needed to jot ideas & make to-do lists."
        "\nFurthermore, Trust is also related to positive praises of security and privacy concerns:"
        "\nIt has advanced privacy protection Joy and Trust can often be confused, as they are contiguous emotions according to Plutchnik's Wheel. In this context, when in doubt, we will apply the following distinction:"
        "\n- If the user makes no explicit mention of their own experience, limiting the feedback to positive aspects and likes, with no involvement of their expectations or their use of the application, then we will consider the review as Joy."
        "\n- If the user makes explicit praise in the review about their own user experience with the application from a personal point of view, including the satisfaction of their goals and needs, then we will consider the review as Trust."
        "\nFor instance, in the following Joy examples:"
        "\n[Example 9] It's very intuitive, and lightweight which makes it a must have for list lovers."
        "\n[Example 10] I like the size options and you can control transparency"
        "\n[Example 11] Does exactly what it needs to in an elegant way and doesnt force monetization or clunky features on you."
        "\nThe author of the review implies positive aspects of the app that they enjoy. However, there is no information about the suitability and acceptance of the app with respect to the personal goals and needs of the user."
        "\nHowever, in the following Trust examples:"
        "\n[Example 12] Best habit tracker I've ever used, free, no ads, data can be exported, overall great."
        "\n[Example 13] Instagram is my app to browse or watch videos and more"
        "\n[Example 14] Use it as my Diary, Really handy ðŸ˜€ ðŸ‘."
        "\nThe author explicitly makes reference to their use of the app, stating their acceptance and reliability and emphasizing positive aspects reinforcing this use."
    ),
    "Anticipation": (
        "Anticipation expresses curiosity and consideration about something unknown or unclear to the user. It implies a sense of alertness and exploration, related to looking ahead to what could come with uncertainty and/or expectation."
        "\nAnticipation can relate to inquiries and questions about particular upcoming changes, also expressed through feature requests or change proposals:"
        "\n[Example 1] Something's I'd have liked to see - get reports of specific projects - ability to add subtasks to tasks"
        "\n[Example 2] I would love a barcode scanner to be included."
        "\nAdditionally, Anticipation also relates to considerations and questions raised by users about something confusing, not clear, for which they do not know how to proceed:"
        "\n[Example 3] But then again, you are not supposed to pause a pomodoro timer, right?"
        "\nWhile they are opposite emotions according to Plutchnik's Wheel, Surprise and Anticipation can often be confused, especially when the sentence implies uncertainty or concerns about something. In this context, when in doubt, we will apply the following distinction:"
        "\n- If the review entails a reactive emotion from the user's point of view (i.e., raised by a particular event or situation, as a reaction to something that happened), then we will consider Surprise."
        "\n- If the review entails a proactive emotion from the user's point of view (i.e., raised by the curiosity and active consideration of the user), then we will consider Anticipation."
        "\nFor instance, in the following Surprise example:"
        "\n[Example 4] wanted to go to an event later in the month showed only 3 attendees but rsvp was closed not showing canceled?"
        "\nThe user experienced an unexpected situation as a result of their use of the app, based on an external incident/event."
        "\nHowever, in the following Anticipation example:"
        "\n[Example 5] I'm trying to figure out what it means when someone phone text messaging an a Goggle account are Linked?"
        "\nWhile the user expresses uncertainty about something, this is not emerging from a reactive situation from a particular event, rather than being a consideration (i.e., anticipation) of the user towards the future of the app."
        "\nFinally, Anticipation can also be confused with sadness when users ask for a missing feature. In this context, when in doubt, we will apply the following distinction:"
        "\n- If the review expresses explicit disappointment with the missing feature, then we will consider Sadness."
        "\n- If the review expresses as a wish for the feature to be included, with no disappointment connotations, then we will consider Anticipation"
    ),
}

EMOTION_EXAMPLES_FEW_SHOOT = {
    "Fear": (
        "[Example 4] #Spyware BEWARE! THEY ARE SPYING ON YOUR TEXTS AND MESSAGES WITH SOFTWARE PROGRAMS AND SUSPENDING ACCOUNTS THE BOTS FIND 'SUSPICIOUS'. They claim to give you total anonymity and they never see messages or calls on their 'encrypted servers' . Then how come the legal department suspended my account if this app is actually private! I had used this app for over 4 years and suddenly as of June 1st, 2022 my businesses work number was completely banished without forewarning. #Spyware Go to Signal..."
        "\n[Example 5] Lacks basic security features and developer seems to lack awareness of terminology: E2E encryption is not really E2E when the data stays unencrypted locally."
        "\n[Example 6] Does not provide transparency for the updates. Most likely selling your personal information."
        "\n[Example 7] Critical to find my folder & document. please improve file system"
        "\n[Example 8] If you even think about privacy, don't install this app at all."
    ),
    "Joy": (
        "[Example 4] Excellent app. If you use Microsoft planner, Outlook mail, share point then it is excellent app for remind you. I love this app and using it daily."
        "\n[Example 5] super very nice gps @come"
        "\n[Example 6] This is exactly what I've been looking for! Customizable and easy to use with basic knowledge."
        "\n[Example 7] Like the app overall but it will not sync with my wife's calendar. We have tried everything we could find online to solve this problem but have been unsuccessful. Please help!"
        "\n[Example 8] Good app but I find some problem in synchronization with my watch and I need to disconnect."
        "\n[Example 9] Its a nice app, it has been working fine for me until this night, its saying theres a problem with the sticker pack and it wont let me send the stickers to whatsapp, and there's no new update for the app, please try and fix this problem"
    ),
    "Anger": (
        "[Example 8] the constant reminder to enable location permissions is just stupid. every single action i do, it has to remind me of enabling the location permission for a pro feature that i won't be using. just leave me alone, i don't want to enable it."
        "\n[Example 9] heaven help you if you try n sync w another email account. crashes CONSTANTLY. hey google nerd-ocracy, your product sucks now, because your priorities suck, and one day you will crash like the androids you have been destroying w this piece of shyte you pass off as an email app. ðŸ–•"
        "\n[Example 10] i personally like that this app can keep you busy, but *i don't like the fact that i can't login to my account unless i pay for the premium version?* LOGGING IN SHOULD BE FREE, NOT SOMETHING THAT PEOPLE SHOULD BE PAYING FOR. why does logging in require paying for the premium version? i don't see the point as to why money should be required for sign-in. ðŸ˜ª"
        "\n[Example 11] Used to work but watch not supported now. No doubt Google play has some issue with China now ffs... Hypocrisy at it's highest level..."
        "\n[Example 12] The watch associated with this app pin on band broke the lobe from base rendering app same value as WRIST RING.. absolute junk. 2wks. Haha"
    ),
    "Sadness": (
        "[Example 6] Can't sync manually imported folders/notebooks on desktop to Android via file system. Local storage has it but Joplin does not include it. Also very old bug 'ERROR ENOENT no such file or directory' still present. Want to like Joplin but support seems prickly or harsh for non devs."
        "\n[Example 7] I think it gets the job done. I can't say that it is my favorite because it isn't necessarily easy to navigate. I had to watch a youtube video to do what I wanted to do with it. It is good for typing notes but I am dissatisfied with the handwriting notetaking portion of the app. Since I use this app for schooling, I was really excited to use it to write my notes but I was disappointed pretty early on. I would say it is good for typing but for a college student, its not the best."
        "\n[Example 8] It just didnt give me the satisfaction of a everyday planner that i like"
        "\n[Example 9] Good app but I find some problem in synchronization with my watch and I need to disconnect."
        "\n[Example 10] I m not satisfied at all. I got the one year premium. How can you charge so much without the ability to sync between devices :O What would happen when i change my phone which i do regulary and how come u cannot save to tje cloud as advertised when the Premium was offered to me. I m looking for an alternative despite having like 11 months left as premium"
        "\n[Example 11] Video calling not mention"
    ),
    "Surprise": (
        "[Example 4] I m not satisfied at all. I got the one year premium. How can you charge so much without the ability to sync between devices :O What would happen when i change my phone which i do regulary and how come u cannot save to tje cloud as advertised when the Premium was offered to me. I m looking for an alternative despite having like 11 months left as premium"
        "\n[Example 5] Good app but I find some problem in synchronization with my watch and I need to disconnect."
        "\n[Example 6] Exactly what I needed! I needed customizable widget for taking notes. I love that it can also do a backup and restore. The only thing I miss is scheduled backups. I setup automatic backup of the whole my phone file system into cloud, but my notes are not backed up automatically, I should manually export them occasionally."
        "\n[Example 7] why does my account isn't eligible for monetization? when i haven't posted anything that could break the community guidelines.fix it"
        "\n[Example 8] I've owned this app for years and have used it regularly. But now I've got a new device and it no longer works. The app hasn't been updated in over three years. Shame because it still has a lot of potential. Perhaps the developer would prefer I pay for a subscription to their web service instead? No thanks, I'd rather have a new version of this app"
    ),
    "Disgust": (
        "[Example 8] Use to be good now its just spam bots and ads"
        "\n[Example 9] I saw that this app had good ratings so I checked it out. It was definitely very cute and easy to use but every second item I added onto a note I get an unskippable 5-10 second ad so that completely ruined the experience for me. Can you imagine any other notetaking app that forces you to wait 10 seconds after you write 2 sentences? Unthinkable for a notetaking app - please stick to banner ads only for these kind of apps. The only reason I'm not giving it 1 star is that it was genuinely cute."
        "\n[Example 10] Decided to uninstall because Todoist is just too buggy, both in the Android app and the web which makes it inconsistent. I used to use the 'share' function on apps to send stuff to Todoist on my phone but an update to Todoist caused the share feature to work only half the time. When it comes to the web, the calendar sync stops working all the time so I've decided it's time for me to find something else that more consistent and reliable. Response to Todoist: Not going to bother. I've done that several times before. Todoist is just in a 'broken' state. You guys always fix one thing only for something else to break."
        "\n[Example 11] It would have been 5 stars and I do not mind paying for features that are useful but putting a majority of the core features, namely bus routes and trains behind a monthly paywall is just greedy, considering these are (likely) the features that people use the most. If a fairer monetization measure is implemented, I will consider returning, however as it stands, I will look for an alternative."
        "\n[Example 12] Total rubbish at keeping connected, pointless for notifications. The old version was good, I have recently updated the app just weeks ago and regret it! I now want to change my watch! Connection is terrible, not getting messages and calls through to my watch, my phone is usually on silence due to night work so I don't bother changing it for a day hence I have got a Huawei band 6 to notify me, has been good for 1 year NOW useless, not connecting back unless done within the app? FIX IT !!!!"
    ),
    "Trust": (
        "[Example 15] The best memo app! Cute, simple and super useful. As a student i can easily manage my tasks by creating seperate folders."
        "\n[Example 16] THE BEST RADAR APP"
        "\n[Example 17] I'm doing a Tour De Habit Trackers and this one is one of the best I've seen. Simple UI, but lets you do stuff like reorder really easily which is a feature that is weirdly missing in *many* of these apps. I need the ability to say 'do xyz every Monday' rather than the more nebulous 'once a week' that Loop allows, and also need to easily be able to do ad-hoc goals on top of my habits, but this app not having those things isn't really a failing since it's got a pretty narrow focus. Great for what it is- better than a lot of the paid options out there."
        "\n[Example 18] This app is best aap for social networking"
        "\n[Example 19] The best app for make stickers for Whatsap. I personally recommend this app for everyone."
    ),
    "Anticipation": (
        "[Example 6] Not 5 Stars yet because it doesn't have a filter to organize the checklists in Alphabetical order or filters to search specific words or notes. It would be very nice to add that feature in a nearly update and then I will pay for the app. It's very good!"
        "\n[Example 7] I love Notebook! I am learning new tips, what can be done, and how to use more if the different features, every day! I will definitely pass the word about how cool Notebook is! I wish there were more similar that would automatically sync across devices! I have way too many notes and use a note app, several times a day! I'm trying to find out if Notepad allows an 'unlimited' amount of notes, or does is stop after we get to a certain amount?"
        "\n[Example 8] Beautiful and Simple. Easier to start up than Toggl. I hope for more powerful features like tagging, a browser app, and deeper analytics in the future. Good luck!"
        "\n[Example 9] App is great. Can we have a theme attached to label, like if we label a note then the theme for that note changes to something that we have assigned to a certain label. Also I hope you launch the text formatting soon"
        "\n[Example 10] This is probably the best notes app for android, clean, straight to the point, easy to use and beautifully designed. The only thing that would make this the perfect app is text formatting."
        "\n[Example 11] A big problem of maps me is that you can not synchronise it whith your other devices so you have to bookmark your interest points on each device separately one by one, that is awful Please activate synchronizing and backuping whith email to work on multi devices for a user Thanks"
    ),
}

# Strategy aliases used by the new package
DEFINITIONS = {
    "zero_shot": EMOTION_DEFINITIONS_ZERO_SHOOT,
    "few_shot_guidelines": EMOTION_DEFINITIONS_FEW_SHOOT,
    "few_shot_guidelines_dataset": {
        e: EMOTION_DEFINITIONS_FEW_SHOOT[e] + "\nAdditional examples include: " + EMOTION_EXAMPLES_FEW_SHOOT.get(e, "")
        for e in EMOTION_DEFINITIONS_FEW_SHOOT
    },
}

STRATEGY_ALIASES = {
    "zeroShoot": "zero_shot",
    "zero_shot": "zero_shot",
    "fewShoot": "few_shot_guidelines",
    "few_shot": "few_shot_guidelines",
    "few_shot_guidelines": "few_shot_guidelines",
    "fewShootExample": "few_shot_guidelines_dataset",
    "fewShootExamples": "few_shot_guidelines_dataset",
    "few_shot_guidelines_dataset": "few_shot_guidelines_dataset",
}
