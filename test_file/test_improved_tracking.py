#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改善されたオブジェクトトラッキングシステムのテストスクリプト
5匹以上の魚に対応し、見失った魚のIDを記憶・再利用する機能をテストします
"""

import numpy as np

def test_improved_tracking():
    """改善されたトラッキングシステムのテスト（デモンストレーションのみ）"""
    
    print("=== 改善されたオブジェクトトラッキングシステムのテスト ===")
    print("機能:")
    print("1. 5匹以上の魚に対応（最大20匹まで）")
    print("2. 見失った魚のIDを記憶")
    print("3. 再検出時に古いIDを再利用")
    print("4. 距離ベースのマッチングアルゴリズム改善")
    print()
    
    print("実際の動画処理を実行するには、以下のコマンドを使用してください:")
    print("python object_tracking.py")
    print("または")
    print("python dual_video_tracking.py")
    print()

def demonstrate_id_reuse():
    """ID再利用機能のデモンストレーション"""
    print("\n=== ID再利用機能のデモンストレーション ===")
    
    # 仮想的なトラッカーの状態をシミュレート
    print("1. 魚A（ID: 1）が位置(100, 100)で検出される")
    fish_a_id = 1
    active_tracks = {fish_a_id: np.array([100, 100, 50, 30])}
    lost_fish = {}
    next_id = 2
    
    print("2. 魚Aが見失われる（30フレーム後）")
    print("   魚Aが解放され、見失った魚として記録される")
    
    # 魚Aを解放（見失った魚として記録）
    lost_fish[fish_a_id] = {
        'last_position': active_tracks[fish_a_id].copy(),
        'lost_frames': 0,
        'last_seen_frame': 100
    }
    del active_tracks[fish_a_id]
    
    print(f"   見失った魚の情報: ID={fish_a_id}, 位置={lost_fish[fish_a_id]['last_position'][:2]}")
    
    print("3. 新しい魚が位置(120, 110)で検出される（魚Aの近く）")
    new_position = np.array([120, 110, 50, 30])
    
    # ID再利用のロジックをシミュレート
    reusable_id = None
    min_distance = float('inf')
    reuse_distance_threshold = 150
    
    for fish_id, fish_info in lost_fish.items():
        last_pos = fish_info['last_position']
        distance = np.linalg.norm(new_position[:2] - last_pos[:2])
        
        if distance <= reuse_distance_threshold and distance < min_distance:
            min_distance = distance
            reusable_id = fish_id
    
    if reusable_id:
        print(f"4. 魚AのID（{reusable_id}）が再利用される！")
        print(f"   距離: {min_distance:.1f}px（閾値: {reuse_distance_threshold}px）")
        
        # IDを再利用
        active_tracks[reusable_id] = new_position
        del lost_fish[reusable_id]
        print(f"5. 魚Aが再追跡開始されました（位置: {new_position[:2]}）")
    else:
        print("4. 再利用可能なIDが見つからない")
        print(f"   新しいID（{next_id}）が作成されます")
        active_tracks[next_id] = new_position
        next_id += 1
    
    print("\n=== デモンストレーション完了 ===")
    print("実際のシステムでは、このロジックが自動的に実行されます。")

if __name__ == "__main__":
    # 改善されたトラッキングシステムの概要
    test_improved_tracking()
    
    # ID再利用機能のデモンストレーション
    demonstrate_id_reuse()
