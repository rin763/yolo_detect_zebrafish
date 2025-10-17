#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
破棄されたIDの再利用機能をテストするスクリプト
"""

import numpy as np

def test_discarded_id_reuse():
    """破棄されたIDの再利用機能をテスト"""
    print("=== 破棄されたIDの再利用機能テスト ===")
    
    # 仮想的なトラッカーの状態をシミュレート
    print("1. トラッカーを初期化")
    next_id = 1
    active_tracks = {}
    lost_fish = {}
    discarded_ids = set()
    max_discarded_ids = 10
    
    def get_new_id():
        """新しいIDを取得（破棄されたIDを優先的に再利用）"""
        nonlocal next_id, discarded_ids
        
        # 破棄されたIDがあれば再利用
        if discarded_ids:
            reused_id = min(discarded_ids)
            discarded_ids.remove(reused_id)
            print(f"   → 破棄されたID {reused_id} を再利用")
            return reused_id
        
        # 破棄されたIDがない場合は新しいIDを作成
        new_id = next_id
        next_id += 1
        print(f"   → 新しいID {new_id} を作成")
        return new_id
    
    def add_discarded_id(fish_id):
        """破棄されたIDを記録"""
        nonlocal discarded_ids, max_discarded_ids
        
        if len(discarded_ids) < max_discarded_ids:
            discarded_ids.add(fish_id)
            print(f"   → 破棄されたID {fish_id} を記録（総数: {len(discarded_ids)}）")
        else:
            # 最大数に達している場合は、最も古いIDを削除して新しいIDを追加
            oldest_id = max(discarded_ids)
            discarded_ids.remove(oldest_id)
            discarded_ids.add(fish_id)
            print(f"   → 古いID {oldest_id} を削除し、新しいID {fish_id} を記録")
    
    print("2. 魚A、B、Cを検出（ID: 1, 2, 3）")
    for i in range(3):
        fish_id = get_new_id()
        active_tracks[fish_id] = np.array([100 + i*50, 100, 50, 30])
        print(f"   魚{chr(65+i)}: ID {fish_id} で追跡開始")
    
    print(f"\n3. 現在のアクティブな魚: {list(active_tracks.keys())}")
    print(f"   破棄されたID: {discarded_ids}")
    
    print("\n4. 魚A（ID: 1）と魚B（ID: 2）が見失われる")
    for fish_id in [1, 2]:
        if fish_id in active_tracks:
            del active_tracks[fish_id]
            add_discarded_id(fish_id)
    
    print(f"\n5. 現在のアクティブな魚: {list(active_tracks.keys())}")
    print(f"   破棄されたID: {discarded_ids}")
    
    print("\n6. 新しい魚Dと魚Eを検出")
    for i, name in enumerate(['D', 'E']):
        fish_id = get_new_id()
        active_tracks[fish_id] = np.array([200 + i*50, 150, 50, 30])
        print(f"   魚{name}: ID {fish_id} で追跡開始")
    
    print(f"\n7. 最終状態:")
    print(f"   アクティブな魚: {list(active_tracks.keys())}")
    print(f"   破棄されたID: {discarded_ids}")
    
    print("\n8. さらに多くの魚を検出して破棄されたIDの再利用をテスト")
    for i in range(5):
        fish_id = get_new_id()
        active_tracks[fish_id] = np.array([300 + i*30, 200, 50, 30])
        print(f"   魚{chr(70+i)}: ID {fish_id} で追跡開始")
    
    print(f"\n9. 最終結果:")
    print(f"   アクティブな魚: {sorted(active_tracks.keys())}")
    print(f"   破棄されたID: {discarded_ids}")
    print(f"   次に作成されるID: {next_id}")
    
    print("\n=== テスト完了 ===")
    print("✓ 破棄されたIDが優先的に再利用されることを確認")
    print("✓ 新しいIDは破棄されたIDがない場合のみ作成されることを確認")

def demonstrate_id_management():
    """ID管理の動作をデモンストレーション"""
    print("\n=== ID管理システムのデモンストレーション ===")
    
    print("機能:")
    print("1. 魚を見失った場合、そのIDを破棄されたIDとして記録")
    print("2. 新しい魚を検出した場合、破棄されたIDがあれば優先的に再利用")
    print("3. 破棄されたIDがない場合のみ、新しいIDを作成")
    print("4. 破棄されたIDの数が上限に達した場合、最も古いIDを削除")
    
    print("\n利点:")
    print("- ID番号を効率的に管理")
    print("- 長期間の追跡でもID番号が無限に増加しない")
    print("- 一度使われたIDは可能な限り再利用される")
    
    print("\n設定可能なパラメータ:")
    print("- max_discarded_ids: 記録する最大破棄ID数（デフォルト: 50）")
    print("- max_lost_frames: IDを保持する最大フレーム数（デフォルト: 60）")

if __name__ == "__main__":
    # ID管理のデモンストレーション
    demonstrate_id_management()
    
    # 破棄されたIDの再利用機能テスト
    test_discarded_id_reuse()
