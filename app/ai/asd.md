
스킬 작성 팁: https://platform.claude.com/docs/ko/agents-and-tools/agent-skills/best-practices

1. 함수 이름(영어) : 명확한 [동사_명사] 형태 (Action + Target)

    ✅ 좋은 예:
        search_bands (밴드 검색)
        get_concert_schedule (공연 일정 조회)
        open_booking_modal (예매 팝업 열기)
        navigate_to_page (페이지 이동)
    ❌ 나쁜 예:
        band_info (동사가 없어서 가져오라는 건지 수정하라는 건지 모름)
        process_data, handle_action (너무 모호해서 AI가 언제 쓸지 모름)

    * 파라미터 타입 힌트 (concert_id: str) 작성 매우 권장


2. Docstring(한국어) : 도구 사용 설명서

    반드시 포함해야 할 3가지:
        무엇을 하는 도구인가? (1줄 요약)
        어떤 상황에서 호출해야 하는가? (호출 조건)
        주의사항이나 제약조건 (금지 사항)
    

3. Return (한국어): 다음 턴에 읽는 새로운 데이터

    원칙 1: 불필요한 데이터 다이어트 (토큰 절약 & 환각 방지)
        DB나 백엔드에서 날아온 100줄짜리 JSON을 그대로 return하지 마세요.
        AI가 답변하는 데 꼭 필요한 필드(이름, 날짜, 가격, 장소)만 추려서 가볍게 return해야 토큰도 아끼고 AI가 헛소리를 안 합니다.
    
    원칙 2: AI에게 행동 지침(힌트)을 함께 주기
        도구 실행 결과와 함께 "message_for_ai" 같은 힌트를 슬쩍 넣어주면 AI가 완벽하게 의도대로 답변합니다.

    원칙 3: 에러 시 파이썬 raise를 던지지 말고 설명 문자열을 return하기
        파이썬 에러가 터지면 대화가 튕깁니다.
        대신 return {"error": "해당 이름의 밴드를 찾지 못했습니다. 유저에게 다시 물어보세요."} 를 넘겨주면, AI가 알아서 "죄송해요, 해당 밴드를 찾지 못했어요. 이름을 다시 확인해 주시겠어요?" 라고 유연하게 대처합니다.

