def get_user_modifications():
    parts = {
        "눈": ["left_eye", "right_eye"],
        "코": ["nose"],
        "입": ["mouth"],
        "광대": ["left_cheek", "right_cheek"],
    }

    print("어느 부위를 수정할까요? (복수 선택, 쉼표로 구분)")
    print("선택 가능:", ", ".join(parts.keys()))
    selected_parts = input("예: 눈,코\n> ").replace(" ", "").split(",")

    modifications = {}
    for part in selected_parts:
        if part not in parts:
            print(f"⚠️ '{part}' 은(는) 인식되지 않습니다. 무시됩니다.")
            continue

        while True:
            scale_text = input(f"{part}을(를) 크게 또는 작게 중 선택해주세요 (크게/작게): ")
            if scale_text in ["크게", "작게"]:
                for region_key in parts[part]:
                    modifications[region_key] = scale_text
                break
            else:
                print("입력 오류. '크게' 또는 '작게'로 입력해주세요.")

    return modifications
