from django.urls import path

from . import views

urlpatterns = [
    path("", views.skills_list, name="agent_skills_list"),
    path("new/", views.skills_create, name="agent_skills_create"),
    path("new/org/", views.skills_create_org, name="agent_skills_create_org"),
    path("<uuid:skill_id>/", views.skills_detail, name="agent_skills_detail"),
    path("<uuid:skill_id>/save/", views.skills_save, name="agent_skills_save"),
    path("<uuid:skill_id>/copy/", views.skills_copy, name="agent_skills_copy"),
    path("<uuid:skill_id>/promote/", views.skills_promote, name="agent_skills_promote"),
    path("<uuid:skill_id>/delete/", views.skills_delete, name="agent_skills_delete"),
    path("<uuid:skill_id>/toggle/", views.skills_toggle, name="agent_skills_toggle"),
]
