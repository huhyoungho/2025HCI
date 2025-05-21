def get_user_modifications():
    parts = ["눈", "코", "입", "귀", "광대", "이마"]
    print("어느 부위를 수정할까요? (복수 선택, 쉼표로 구분)")
    print("선택 가능:", ", ".join(parts))
    selected_parts = input("예: 눈,코\n> ").replace(" ", "").split(",")

    modifications = {}
    for part in selected_parts:
        if part in parts:
            scale = input(f"{part}을(를) 크게 또는 작게 중 선택해주세요 (크게/작게): ")
            if scale in ["크게", "작게"]:
                modifications[part] = scale
    return modifications

# modifications = get_user_modifications()
# 예: {'눈': '크게', '입': '작게'}
