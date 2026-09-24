action2prompt = {
    "Deny": "You should directly refuse to admit your behavior is problematic or needs change.",
    "Downplay": "You should downplay the importance or impact of your behavior.",
    "Blame": "You should blame external factors or others to justify your behavior.",
    "Inform": "You should share details about your background, experiences, or emotions revealing the current state.",
    "Engage": "You should interact with counselor consistently based on your state and mimic the style in the reference conversation.",
    "Hesitate": "You should show uncertainty, indicating ambivalence about change.",
    "Doubt": "You should expresse skepticism about the practicality or success of proposed changes but not reveal further information.",
    "Acknowledge": "You should acknowledge the need for change.",
    "Accept": "You should agree to adopt the suggested action plan.",
    "Reject": "You should decline the proposed plan, deeming it unsuitable.",
    "Plan": "You should propose or detail steps for a change plan.",
    "Terminate": "You should highlight current state and engagement, express a desire to end the current session, and suggest further discussion be deferred to a later time.",
}

def state2prompt(behavior, goal):
    return {
        "Precontemplation": f"You doesn't think your {behavior} is problematic and wants to sustain.",
        "Contemplation": f"You feels that your {behavior} is problematic, but still hesitate about {goal}.",
        "Preparation": f"You gets ready to take action to change and begins discuss about steps toward {goal}.",
    }

def topic2description(behavior, goal):
    return {
        "Infection": f"When discussing [topic], The counselor may discuss how {behavior} can weaken your immune system, making you more susceptible to infections. They may also talk about how {goal} can help strengthen your immune system and reduce your risk of frequent or severe infections.",
        "Hypertension": f"When discussing [topic], The counselor may explore how {behavior} can contribute to increased blood pressure and raise your risk of hypertension. They may also discuss how {goal} can help maintain a healthier blood pressure level and lower your risk of complications from hypertension.",
        "Flu": f"When discussing [topic], The counselor may explain how {behavior} can increase your chances of contracting the flu or experiencing more severe symptoms. They may also highlight how {goal} can improve your immune defense, reducing your likelihood of catching the flu or lessening its impact.",
        "Inflammation": f"When discussing [topic], The counselor may address how {behavior} can lead to chronic inflammation in your body, increasing your risk for related health conditions. They may also discuss how {goal} can help reduce inflammation and promote better long-term health.",
        "Liver Disease": f"When discussing [topic], The counselor may explore how {behavior} can cause liver damage, increasing your risk of liver disease. They may also discuss how {goal} can protect your liver, prevent damage, and lower your chances of developing serious liver conditions.",
        "Lung Cancer": f"When discussing [topic], The counselor may discuss how {behavior} can increase your risk of developing lung cancer. They may also highlight how {goal} can reduce your risk of lung cancer and improve your overall respiratory health.",
        "COPD": f"When discussing [topic], The counselor may explore how {behavior} can worsen symptoms of COPD, making it harder for you to breathe. They may also explain how {goal} can improve your lung function and help manage COPD symptoms.",
        "Asthma": f"When discussing [topic], The counselor may explain how {behavior} can trigger or worsen asthma attacks. They may also discuss how {goal} can help control asthma symptoms and improve your ability to manage the condition.",
        "Stroke": f"When discussing [topic], The counselor may address how {behavior} can increase your risk of stroke by negatively impacting your cardiovascular health. They may also highlight how {goal} can help reduce this risk and support a healthier cardiovascular system.",
        "Diabetes": f"When discussing [topic], The counselor may explore how {behavior} can contribute to the development or worsening of diabetes. They may also discuss how {goal} can help manage your blood sugar levels and reduce the risk of diabetes-related complications.",
        "Physical Activity": f"When discussing [topic], The counselor may discuss how {behavior} can reduce your physical activity levels, increasing your risk of various health problems. They may also explain how {goal} can boost your physical activity, improving your fitness and overall health.",
        "Sport": f"When discussing [topic], The counselor may explain how {behavior} can decrease your performance or participation in sports. They may also discuss how {goal} can enhance your physical conditioning and boost your ability to engage in sporting activities.",
        "Physical Fitness": f"When discussing [topic], The counselor may discuss how {behavior} can negatively affect your physical fitness, reducing your overall health. They may also explain how {goal} can improve your fitness level and contribute to better health and well-being.",
        "Strength": f"When discussing [topic], The counselor may explore how {behavior} can lead to a loss of strength, making everyday tasks more difficult. They may also discuss how {goal} can help you build or regain muscle strength.",
        "Flexibility": f"When discussing [topic], The counselor may explain how {behavior} can reduce your flexibility, causing stiffness or discomfort. They may also highlight how {goal} can improve your flexibility, making movement easier and reducing the risk of injury.",
        "Endurance": f"When discussing [topic], The counselor may discuss how {behavior} can lower your endurance, making it harder to sustain physical activities for long periods. They may also explore how {goal} can improve your stamina and overall physical endurance.",
        "Dentistry": f"When discussing [topic], The counselor may address how {behavior} can lead to poor dental hygiene, increasing your risk of cavities or gum disease. They may also explain how {goal} can improve your oral health and reduce the likelihood of dental problems.",
        "Caregiver Burden": f"When discussing [topic], The counselor may explore how {behavior} can place additional stress on caregivers, leading to burnout or reduced quality of care. They may also discuss how {goal} can alleviate the burden on caregivers and improve the care provided.",
        "Independent Living": f"When discussing [topic], The counselor may explain how {behavior} can limit your ability to live independently, making you more reliant on others. They may also discuss how {goal} can help you regain independence and improve your quality of life.",
        "Human Appearance": f"When discussing [topic], The counselor may address how {behavior} can negatively impact your appearance, such as causing skin issues or weight gain. They may also explain how {goal} can improve your physical appearance and boost your self-confidence.",
        "Depression": f"When discussing [topic], The counselor may explore how {behavior} can worsen symptoms of depression, affecting your mood and daily life. They may also discuss how {goal} can improve your emotional well-being and reduce the impact of depression.",
        "Chronodisruption": f"When discussing [topic], The counselor may explain how {behavior} can disrupt your body’s natural rhythms, leading to sleep problems or fatigue. They may also highlight how {goal} can restore healthy sleep patterns and improve your overall well-being.",
        "Anxiety Disorders": f"When discussing [topic], The counselor may explore how {behavior} can increase your anxiety, leading to stress and panic attacks. They may also discuss how {goal} can help manage anxiety and promote emotional stability.",
        "Cognitive Decline": f"When discussing [topic], The counselor may discuss how {behavior} can accelerate cognitive decline, affecting your memory and thinking abilities. They may also highlight how {goal} can help protect brain function and slow cognitive deterioration.",
        "Safe Sex": f"When discussing [topic], The counselor may explain how {behavior} can increase your risk of sexually transmitted infections or unintended pregnancies. They may also discuss how {goal} can promote safer sexual practices and reduce these risks.",
        "Maternal Health": f"When discussing [topic], The counselor may address how {behavior} can negatively affect your health during pregnancy, increasing the risk of complications. They may also discuss how {goal} can support a healthier pregnancy and reduce the likelihood of problems during childbirth.",
        "Preterm Birth": f"When discussing [topic], The counselor may explore how {behavior} can increase the risk of preterm birth, leading to complications for both mother and baby. They may also highlight how {goal} can help ensure a full-term, healthy pregnancy.",
        "Miscarriage": f"When discussing [topic], The counselor may explain how {behavior} can increase the risk of miscarriage, causing emotional and physical distress. They may also discuss how {goal} can help support a healthy pregnancy and reduce miscarriage risk.",
        "Birth Defects": f"When discussing [topic], The counselor may address how {behavior} can raise the risk of birth defects during pregnancy. They may also explain how {goal} can promote a healthier pregnancy and reduce the risk of complications.",
        "Productivity": f"When discussing [topic], The counselor may explore how {behavior} can negatively affect your productivity at work, making it harder to perform well. They may also discuss how {goal} can improve your focus and efficiency.",
        "Absenteeism": f"When discussing [topic], The counselor may explain how {behavior} can increase absenteeism, causing you to miss work or other responsibilities. They may also highlight how {goal} can help you be more consistent and present in your daily life.",
        "Workplace Relationships": f"When discussing [topic], The counselor may discuss how {behavior} can strain relationships with colleagues, leading to conflicts at work. They may also explain how {goal} can help improve communication and foster positive workplace relationships.",
        "Career Break": f"When discussing [topic], The counselor may address how {behavior} can lead to career interruptions or breaks, affecting your professional progress. They may also discuss how {goal} can help you maintain continuity in your career.",
        "Career Assessment": f"When discussing [topic], The counselor may explain how {behavior} can affect career assessments, leading to negative evaluations or feedback. They may also highlight how {goal} can improve your performance and result in more favorable assessments.",
        "Absence Rate": f"When discussing [topic], The counselor may explore how {behavior} can increase your absence rate at work, impacting your job security. They may also discuss how {goal} can help you reduce absences and improve your work attendance.",
        "Salary": f"When discussing [topic], The counselor may address how {behavior} can hinder salary progression, limiting your earning potential. They may also discuss how {goal} can help you increase your salary and financial stability.",
        "Workplace Wellness": f"When discussing [topic], The counselor may discuss how {behavior} can undermine workplace wellness initiatives, affecting your health at work. They may also explain how {goal} can help you benefit from wellness programs and improve your job satisfaction.",
        "Workplace Incivility": f"When discussing [topic], The counselor may explore how {behavior} can contribute to incivility in the workplace, leading to a negative environment. They may also highlight how {goal} can help promote respect and cooperation in your work setting.",
        "Cost of Living": f"When discussing [topic], The counselor may discuss how {behavior} can make it harder to manage the cost of living, leading to financial strain. They may also explain how {goal} can help improve your financial situation and reduce stress related to expenses.",
        "Personal Budget": f"When discussing [topic], The counselor may address how {behavior} can make it difficult to stick to a personal budget, leading to financial instability. They may also discuss how {goal} can help you manage your finances more effectively.",
        "Debt": f"When discussing [topic], The counselor may explain how {behavior} can lead to increased debt, affecting your financial security. They may also discuss how {goal} can help you reduce debt and improve your financial well-being.",
        "Income Deficit": f"When discussing [topic], The counselor may explore how {behavior} can contribute to income deficits, making it harder to cover basic expenses. They may also explain how {goal} can help improve your financial management and income stability.",
        "Family Estrangement": f"When discussing [topic], The counselor may discuss how {behavior} can lead to emotional or physical distance between family members. They may also highlight how {goal} can help repair family relationships and promote reconciliation.",
        "Family Disruption": f"When discussing [topic], The counselor may explain how {behavior} can disrupt family dynamics, leading to conflict or instability. They may also discuss how {goal} can strengthen family bonds and promote harmony.",
        "Divorce": f"When discussing [topic], The counselor may address how {behavior} can contribute to marital conflict, potentially leading to divorce. They may also discuss how {goal} can improve communication and reduce the risk of divorce.",
        "Role Model": f"When discussing [topic], The counselor may explain how {behavior} can affect your ability to serve as a positive role model, particularly for your children. They may also highlight how {goal} can help you set a better example for those who look up to you.",
        "Child Development": f"When discussing [topic], The counselor may discuss how {behavior} can impact your child’s emotional, social, or cognitive development. They may also explain how {goal} can support healthier growth and development in your child.",
        "Paternal Bond": f"When discussing [topic], The counselor may explore how {behavior} can weaken the bond between you and your child, affecting your relationship. They may also discuss how {goal} can help strengthen this bond and promote a closer connection.",
        "Child Care": f"When discussing [topic], The counselor may explain how {behavior} can interfere with your ability to provide consistent child care. They may also highlight how {goal} can help you offer more stable and nurturing care for your child.",
        "Habituation": f"When discussing [topic], The counselor may discuss how {behavior} can negatively affect a child’s ability to develop healthy habits or adapt to new environments. They may also explain how {goal} can support positive habituation and learning.",
        "Arrest": f"When discussing [topic], The counselor may address how {behavior} can increase your risk of arrest, leading to legal trouble. They may also highlight how {goal} can help you avoid legal issues and stay on the right side of the law.",
        "Imprisonment": f"When discussing [topic], The counselor may explain how {behavior} can lead to imprisonment, causing long-term legal and social consequences. They may also discuss how {goal} can help you avoid incarceration and promote responsible behavior.",
        "Child Custody": f"When discussing [topic], The counselor may explore how {behavior} can affect your ability to maintain child custody, leading to legal challenges. They may also discuss how {goal} can help you improve your parenting abilities and secure your custody rights.",
        "Traffic Ticket": f"When discussing [topic], The counselor may address how {behavior} can increase your chances of receiving traffic tickets or other legal penalties. They may also discuss how {goal} can help you adopt safer driving practices and avoid infractions.",
        "Complaint": f"When discussing [topic], The counselor may explain how {behavior} can result in legal complaints or disputes. They may also highlight how {goal} can help you reduce conflicts and maintain positive relationships with others.",
        "Attendance": f"When discussing [topic], The counselor may explore how {behavior} can affect your attendance, causing you to miss school or other responsibilities. They may also discuss how {goal} can help you improve your attendance and stay on track.",
        "Suspension": f"When discussing [topic], The counselor may explain how {behavior} can lead to suspension from school, affecting your academic progress. They may also highlight how {goal} can help you avoid disciplinary actions and stay engaged in school.",
        "Scholarship": f"When discussing [topic], The counselor may address how {behavior} can reduce your eligibility for scholarships, limiting academic opportunities. They may also discuss how {goal} can help improve your academic performance and increase your chances of receiving scholarships.",
        "Exam": f"When discussing [topic], The counselor may discuss how {behavior} can negatively impact your exam preparation and performance, leading to lower grades. They may also explain how {goal} can help improve your focus and exam results.",
        "Health": f"When discussing [topic], The counselor may talk about how {behavior} affects your overall health, including physical, mental, and emotional well-being. They may focus on specific health conditions, fitness levels, mental health, or healthcare practices. The counselor can also discuss how {goal} can improve your health and reduce risks.",
        "Diseases": f"When discussing [topic], The counselor may talk about how {behavior} leads to specific health conditions, such as infections, hypertension, flu, inflammation, liver disease, lung cancer, chronic obstructive pulmonary disease (COPD), asthma, stroke, and diabetes. The counselor can also discuss how {goal} can help prevent or manage these diseases.",
        "Fitness": f"When discussing [topic], The counselor may talk about how {behavior} reduces your physical activity, strength, flexibility, or endurance. The counselor can also discuss how {goal} can improve your fitness and overall physical health.",
        "Health Care": f"When discussing [topic], The counselor may talk about how {behavior} affects your ability to maintain personal healthcare, such as dental hygiene, caregiver burden, independent living, or human appearance. The counselor can also discuss how {goal} can improve your self-care and healthcare routines.",
        "Mental Disorders": f"When discussing [topic], The counselor may talk about how {behavior} contributes to mental health issues, such as depression, chronodisruption, anxiety disorders, or cognitive decline. The counselor can also discuss how {goal} can improve your emotional well-being and cognitive health.",
        "Sexual Health": f"When discussing [topic], The counselor may talk about how {behavior} affects your sexual and reproductive health, including safe sex practices, maternal health, preterm birth, miscarriage, or birth defects. The counselor can also discuss how {goal} can help reduce risks and support a healthier sexual lifestyle.",
        "Economy": f"When discussing [topic], The counselor may talk about how {behavior} impacts your financial stability, job performance, or overall economic well-being. They may discuss issues like work productivity, financial management, or income deficits. The counselor can also discuss how {goal} can improve your financial situation and career success.",
        "Employment": f"When discussing [topic], The counselor may talk about how {behavior} affects your job performance, including productivity, absenteeism, workplace relationships, career breaks, career assessments, absence rates, salary, workplace wellness, or workplace incivility. The counselor can also discuss how {goal} can improve your professional performance and career progression.",
        "Personal Finance": f"When discussing [topic], The counselor may talk about how {behavior} impacts your personal finances, including budgeting, debt management, cost of living, or income deficits. The counselor can also discuss how {goal} can improve your financial stability and reduce financial stress.",
        "Interpersonal Relationships": f"When discussing [topic], The counselor may talk about how {behavior} affects your relationships with family, children, or others. They may focus on how family dynamics, parenting roles, or conflicts are influenced by your actions. The counselor can also discuss how {goal} can help strengthen relationships and foster positive connections.",
        "Family": f"When discussing [topic], The counselor may talk about how {behavior} leads to family estrangement, family disruption, or divorce. The counselor can also discuss how {goal} can help resolve family conflicts and improve relationships.",
        "Parenting": f"When discussing [topic], The counselor may talk about how {behavior} affects your parenting style and your child’s development, including role modeling, paternal bonding, child care, or habituation. The counselor can also discuss how {goal} can improve your parenting and support your child’s growth.",
        "Law": f"When discussing [topic], The counselor may talk about how {behavior} may lead to legal issues, such as arrests, traffic violations, or family law disputes. They may focus on the consequences of these actions and the risks involved. The counselor can also discuss how {goal} can help you avoid legal trouble and promote a law-abiding lifestyle.",
        "Criminal Law": f"When discussing [topic], The counselor may talk about how {behavior} can lead to legal consequences, such as arrest, imprisonment, or complaints. The counselor can also discuss how {goal} can help avoid these situations and promote responsible behavior.",
        "Family Law": f"When discussing [topic], The counselor may talk about how {behavior} affects family law matters, such as child custody disputes. The counselor can also discuss how {goal} can improve your parenting and strengthen your legal position.",
        "Traffic Law": f"When discussing [topic], The counselor may talk about how {behavior} increases your chances of receiving traffic tickets or fines. The counselor can also discuss how {goal} can help you drive responsibly and avoid future violations.",
        "Education": f"When discussing [topic], The counselor may talk about how {behavior} affects your academic performance and opportunities, such as attendance, exam results, or scholarship eligibility. They may focus on how these behaviors hinder your educational success. The counselor can also discuss how {goal} can help you improve your academic performance and stay engaged in school.",
        "Student Affairs": f"When discussing [topic], The counselor may talk about how {behavior} impacts your attendance, causes suspensions, or affects your eligibility for scholarships. The counselor can also discuss how {goal} can help improve your participation and academic success.",
        "Academic Achievement": f"When discussing [topic], The counselor may talk about how {behavior} affects your academic performance or productivity in study. The counselor can also discuss how {goal} can help you achieve better grades and academic success.",
    }

topic_graph = {
    "Health": {
        "Mental Disorders": 2,
        "Diseases": 2,
        "Sexual Health": 2,
        "Fitness": 2,
        "Health Care": 2,
        "Workplace Wellness": 1,
        "Interpersonal Relationships": 3,
        "Law": 3,
        "Economy": 3,
        "Education": 3,
    },
    "Interpersonal Relationships": {
        "Parenting": 2,
        "Family": 2,
        "Health": 3,
        "Law": 3,
        "Economy": 3,
        "Education": 3,
        "Workplace Relationships": 1,
    },
    "Law": {
        "Child Custody": 1,
        "Traffic Ticket": 1,
        "Criminal Law": 2,
        "Health": 3,
        "Interpersonal Relationships": 3,
        "Economy": 3,
        "Education": 3,
    },
    "Economy": {
        "Personal Finance": 2,
        "Employment": 2,
        "Family": 3,
        "Law": 3,
        "Health": 3,
        "Interpersonal Relationships": 3,
        "Education": 3,
    },
    "Education": {
        "Academic Achievement": 1,
        "Student Affairs": 2,
        "Exam": 1,
        "Health": 3,
        "Interpersonal Relationships": 3,
        "Law": 3,
        "Economy": 3,
    },
    "Student Affairs": {
        "Attendance": 1,
        "Suspension": 1,
        "Scholarship": 1,
        "Education": 2,
    },
    "Exam": {
        "Education": 2,
    },
    "Academic Achievement": {
        "Education": 2,
    },
    "Scholarship": {
        "Student Affairs": 1,
    },
    "Suspension": {
        "Student Affairs": 1,
    },
    "Attendance": {
        "Student Affairs": 1,
    },
    "Employment": {
        "Economy": 2,
        "Productivity": 1,
        "Absenteeism": 1,
        "Career Assessment": 1,
        "Absence Rate": 1,
        "Career Break": 1,
        "Salary": 1,
        "Workplace Wellness": 1,
        "Workplace Relationships": 1,
        "Workplace Incivility": 1,
    },
    "Productivity": {
        "Employment": 1,
    },
    "Absenteeism": {
        "Employment": 1,
    },
    "Career Assessment": {
        "Employment": 1,
    },
    "Absence Rate": {
        "Employment": 1,
    },
    "Career Break": {
        "Employment": 1,
    },
    "Salary": {
        "Employment": 1,
    },
    "Workplace Wellness": {
        "Employment": 1,
        "Health": 1,
    },
    "Workplace Relationships": {
        "Employment": 1,
        "Interpersonal Relationships": 1,
    },
    "Workplace Incivility": {
        "Employment": 1,
    },
    "Personal Finance": {
        "Economy": 2,
        "Debt": 1,
        "Cost of Living": 1,
        "Income Deficit": 1,
        "Personal Budget": 1,
    },
    "Debt": {
        "Personal Finance": 1,
    },
    "Cost of Living": {
        "Personal Finance": 1,
    },
    "Income Deficit": {
        "Personal Finance": 1,
    },
    "Personal Budget": {
        "Personal Finance": 1,
    },
    "Criminal Law": {
        "Law": 2,
        "Arrest": 1,
        "Complaint": 1,
        "Imprisonment": 1,
    },
    "Arrest": {
        "Criminal Law": 1,
    },
    "Complaint": {
        "Criminal Law": 1,
    },
    "Imprisonment": {
        "Criminal Law": 1,
    },
    "Traffic Ticket": {
        "Law": 1,
    },
    "Child Custody": {
        "Law": 1,
        "Family": 1,
        "Parenting": 1,
    },
    "Family": {
        "Interpersonal Relationships": 2,
        "Family Estrangement": 1,
        "Family Disruption": 1,
        "Divorce": 1,
        "Child Custody": 1,
    },
    "Family Estrangement": {
        "Family": 1,
    },
    "Family Disruption": {
        "Family": 1,
    },
    "Divorce": {
        "Family": 1,
    },
    "Parenting": {
        "Interpersonal Relationships": 2,
        "Role Model": 1,
        "Child Development": 1,
        "Paternal Bond": 1,
        "Child Care": 1,
        "Habituation": 1,
        "Child Custody": 1,
    },
    "Role Model": {
        "Parenting": 1,
    },
    "Child Development": {
        "Parenting": 1,
    },
    "Paternal Bond": {
        "Parenting": 1,
    },
    "Child Care": {
        "Parenting": 1,
        "Health Care": 1,
    },
    "Habituation": {
        "Parenting": 1,
    },
    "Health Care": {
        "Health": 2,
        "Dentistry": 1,
        "Caregiver Burden": 1,
        "Independent Living": 1,
        "Human Appearance": 1,
        "Child Care": 1,
        "Maternal Health": 1,
    },
    "Dentistry": {
        "Health Care": 1,
    },
    "Caregiver Burden": {
        "Health Care": 1,
    },
    "Independent Living": {
        "Health Care": 1,
    },
    "Human Appearance": {
        "Health Care": 1,
    },
    "Fitness": {
        "Health": 2,
        "Strength": 1,
        "Flexibility": 1,
        "Endurance": 1,
        "Sport": 1,
        "Physical Activity": 1,
        "Physical Fitness": 1,
    },
    "Strength": {
        "Fitness": 1,
    },
    "Flexibility": {
        "Fitness": 1,
    },
    "Endurance": {
        "Fitness": 1,
    },
    "Sport": {
        "Fitness": 1,
    },
    "Physical Activity": {
        "Fitness": 1,
    },
    "Physical Fitness": {
        "Fitness": 1,
    },
    "Sexual Health": {
        "Health": 2,
        "Maternal Health": 1,
        "Miscarriage": 1,
        "Safe Sex": 1,
        "Birth Defects": 1,
        "Preterm Birth": 1,
    },
    "Maternal Health": {
        "Sexual Health": 1,
        "Health Care": 1,
    },
    "Miscarriage": {
        "Sexual Health": 1,
    },
    "Safe Sex": {
        "Sexual Health": 1,
    },
    "Birth Defects": {
        "Sexual Health": 1,
    },
    "Preterm Birth": {
        "Sexual Health": 1,
    },
    "Mental Disorders": {
        "Health": 2,
        "Depression": 1,
        "Chronodisruption": 1,
        "Anxiety Disorders": 1,
        "Cognitive Decline": 1,
    },
    "Depression": {
        "Mental Disorders": 1,
    },
    "Chronodisruption": {
        "Mental Disorders": 1,
    },
    "Anxiety Disorders": {
        "Mental Disorders": 1,
    },
    "Cognitive Decline": {
        "Mental Disorders": 1,
    },
    "Diseases": {
        "Health": 2,
        "Infection": 1,
        "Hypertension": 1,
        "Flu": 1,
        "Inflammation": 1,
        "Liver Disease": 1,
        "Lung Cancer": 1,
        "COPD": 1,
        "Asthma": 1,
        "Stroke": 1,
        "Diabetes": 1,
    },
    "Infection": {
        "Diseases": 1,
    },
    "Hypertension": {
        "Diseases": 1,
    },
    "Flu": {
        "Diseases": 1,
    },
    "Inflammation": {
        "Diseases": 1,
    },
    "Liver Disease": {
        "Diseases": 1,
    },
    "Lung Cancer": {
        "Diseases": 1,
    },
    "COPD": {
        "Diseases": 1,
    },
    "Asthma": {
        "Diseases": 1,
    },
    "Stroke": {
        "Diseases": 1,
    },
    "Diabetes": {
        "Diseases": 1,
    },
}